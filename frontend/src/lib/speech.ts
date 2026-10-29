export interface TTTOptions {
  voice?: string;
  ssml?: boolean;
  prosody?: {
    rate?: number;
    pitch?: number;
    volume?: number;
  };
}

export interface STTResult {
  transcript: string;
  isFinal: boolean;
  confidence?: number;
  latencyMs: number;
}

const SAMPLE_RATE_22KHZ = 22050;

export class PiperTTSClient {
  private apiUrl: string;
  private audioContext: AudioContext | null = null;
  private currentSource: AudioBufferSourceNode | null = null;
  private currentGain: GainNode | null = null;
  private latencyHistory: number[] = [];
  private bargeInCallbacks: Array<() => void> = [];

  constructor(apiUrl: string = "/api") {
    this.apiUrl = apiUrl;
  }

  private ensureAudioContext(): AudioContext {
    if (!this.audioContext) {
      const AudioContextCtor = window.AudioContext || (window as any).webkitAudioContext;
      this.audioContext = new AudioContextCtor({
        latencyHint: "interactive" as const,
        sampleRate: SAMPLE_RATE_22KHZ,
      });
    }
    return this.audioContext;
  }

  stop(): void {
    if (this.currentSource) {
      try {
        this.currentSource.stop();
      } catch {
      }
      this.currentSource.disconnect();
      this.currentSource = null;
    }
    if (this.currentGain) {
      this.currentGain.disconnect();
      this.currentGain = null;
    }
  }

  isPlaying(): boolean {
    return this.currentSource !== null;
  }

  onBargeIn(callback: () => void): void {
    this.bargeInCallbacks.push(callback);
  }

  private triggerBargeIn(): void {
    this.stop();
    this.bargeInCallbacks.forEach((cb) => cb());
  }

  async speak(text: string, options: TTTOptions = {}): Promise<void> {
    const firstAudioStart = performance.now();

    const payload: Record<string, unknown> = {
      text,
      voice: options.voice || "en_US-less-medium",
      ssml: options.ssml || false,
    };

    if (options.ssml && options.prosody) {
      const prosodyAttrs: string[] = [];
      if (options.prosody.rate !== undefined) {
        prosodyAttrs.push(`rate="${options.prosody.rate}"`);
      }
      if (options.prosody.pitch !== undefined) {
        prosodyAttrs.push(`pitch="${options.prosody.pitch}"`);
      }
      if (options.prosody.volume !== undefined) {
        prosodyAttrs.push(`volume="${options.prosody.volume}"`);
      }
      if (prosodyAttrs.length > 0) {
        const prosodyOpen = `<prosody ${prosodyAttrs.join(" ")}>`;
        payload.text = `<speak>${prosodyOpen}${text}</prosody></speak>`;
        payload.ssml = true;
      }
    }

    const response = await fetch(`${this.apiUrl}/tts`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(`TTS error: ${error.detail || response.statusText}`);
    }

    const arrayBuffer = await response.arrayBuffer();

    const audioContext = this.ensureAudioContext();

    if (audioContext.state === "suspended") {
      await audioContext.resume();
    }

    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

    this.currentSource = audioContext.createBufferSource();
    this.currentGain = audioContext.createGain();

    this.currentGain.gain.value = 1.0;

    this.currentSource.connect(this.currentGain);
    this.currentGain.connect(audioContext.destination);

    this.currentSource.buffer = audioBuffer;
    this.currentSource.start(0);

    const firstAudioLatency = performance.now() - firstAudioStart;
    this.latencyHistory.push(firstAudioLatency);

    return new Promise<void>((resolve) => {
      if (this.currentSource) {
        this.currentSource.onended = () => {
          this.currentSource = null;
          this.currentGain = null;
          resolve();
        };
      } else {
        resolve();
      }
    });
  }

  getFirstAudioLatency(): number {
    if (this.latencyHistory.length === 0) return 0;
    const sum = this.latencyHistory.reduce((a, b) => a + b, 0);
    return sum / this.latencyHistory.length;
  }
}

export class WebSpeechSTT {
  private recognition: SpeechRecognition | null = null;
  private interimCallback: ((result: STTResult) => void) | null = null;
  private finalCallback: ((result: STTResult) => void) | null = null;
  private silenceTimeoutMs: number = 1500;
  private silenceTimer: ReturnType | null = null;
  private lastTranscript: string = "";
  private startTime: number = 0;
  private silenceCommitTimer: ReturnType | null = null;

  constructor(silenceTimeoutMs: number = 1500) {
    this.silenceTimeoutMs = silenceTimeoutMs;

    if (
      typeof window !== "undefined" &&
      ("SpeechRecognition" in window || "webkitSpeechRecognition" in window)
    ) {
      const SpeechRecognitionAPI =
        (window as any).SpeechRecognition ||
        (window as any).webkitSpeechRecognition;
      this.recognition = new SpeechRecognitionAPI();
      this.recognition.continuous = true;
      this.recognition.interimResults = true;
      this.recognition.lang = "en-US";

      this.recognition.onresult = (event: any) => {
        const now = performance.now();
        const latencyMs = now - this.startTime;

        let interimText = "";
        let finalText = "";

        for (
          let i = this.recognition!.resultIndex;
          i < event.results.length;
          i++
        ) {
          const transcript = event.results[i][0].transcript;

          if (event.results[i].isFinal) {
            finalText += transcript;
          } else {
            interimText += transcript;
          }
        }

        if (finalText) {
          this.lastTranscript += finalText;
          if (this.finalCallback) {
            this.finalCallback({
              transcript: finalText,
              isFinal: true,
              confidence: event.results[0][0].confidence,
              latencyMs,
            });
          }
        }

        if (interimText && this.interimCallback) {
          this.interimCallback({
            transcript: interimText,
            isFinal: false,
            latencyMs,
          });
        }

        if (this.silenceTimer) {
          clearTimeout(this.silenceTimer);
        }
        this.silenceTimer = setTimeout(() => {
          if (this.finalCallback) {
            this.finalCallback({
              transcript: this.lastTranscript,
              isFinal: true,
              latencyMs: 0,
            });
            this.lastTranscript = "";
          }
          this.commitSilence();
        }, this.silenceTimeoutMs);
      };

      this.recognition.onerror = (event: any) => {
        console.error("STT error:", event.error);
      };

      this.recognition.onend = () => {
        if (this.silenceCommitTimer) {
          clearTimeout(this.silenceCommitTimer);
          this.silenceCommitTimer = null;
        }
      };
    }
  }

  start(
    interimCallback: (result: STTResult) => void,
    finalCallback: (result: STTResult) => void
  ): void {
    if (!this.recognition) {
      throw new Error("SpeechRecognition not supported in this browser");
    }

    this.interimCallback = interimCallback;
    this.finalCallback = finalCallback;
    this.startTime = performance.now();

    this.recognition.start();
  }

  private commitSilence(): void {
    if (this.lastTranscript) {
      this.lastTranscript = "";
    }
  }

  stop(): void {
    if (this.recognition) {
      this.recognition.stop();
    }
    if (this.silenceTimer) {
      clearTimeout(this.silenceTimer);
      this.silenceTimer = null;
    }
  }

  abort(): void {
    this.stop();
  }

  isSupported(): boolean {
    return !!this.recognition;
  }
}

export const speakPiper = async (
  text: string,
  options: TTTOptions = {},
  bargeInHandler?: () => void
): Promise<PiperTTSClient> => {
  const client = new PiperTTSClient();

  if (bargeInHandler) {
    client.onBargeIn(bargeInHandler);

    const handleBargeIn = (event: Event) => {
      const isUserSpeech = event.type === "keydown" || event.type === "mousedown";
      if (client.isPlaying() && isUserSpeech) {
        client.stop();
        if (bargeInHandler) bargeInHandler();
      }
    };

    document.addEventListener("keydown", handleBargeIn);
    document.addEventListener("mousedown", handleBargeIn);

    setTimeout(() => {
      document.removeEventListener("keydown", handleBargeIn);
      document.removeEventListener("mousedown", handleBargeIn);
    }, 30000);
  }

  await client.speak(text, options);
  return client;
};

export const startSTT = (
  onInterim: (result: STTResult) => void,
  onFinal: (result: STTResult) => void,
  silenceTimeoutMs: number = 1500
): WebSpeechSTT => {
  const stt = new WebSpeechSTT(silenceTimeoutMs);
  stt.start(onInterim, onFinal);
  return stt;
};

export const stopSTT = (stt: WebSpeechSTT): void => {
  stt.stop();
};
