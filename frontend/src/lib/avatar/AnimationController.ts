import { useEffect, useRef, useState, useCallback } from 'react'
import { useFrame } from '@react-three/fiber'
import { useGLTF, useAnimations } from '@react-three/drei'
import { AnimationMixer, AnimationClip, LoopRepeat, Group, Bone, Vector3 } from 'three'
import { create } from 'zustand'

interface MediaPipeLandmark {
  x: number
  y: number
  z: number
  visibility?: number
}

interface MediaPipeLandmarks {
  [key: string]: MediaPipeLandmark
}

interface AnimationState {
  currentClip: string | null
  isPlaying: boolean
  time: number
  weights: Record<string, number>
}

interface AnimationControllerStore extends AnimationState {
  controller: AnimationController | null
  play: (clipName: string, fadeTime?: number) => void
  stop: (fadeTime?: number) => void
  blend: (clipName: string, weight: number, fadeTime?: number) => void
  setTime: (time: number) => void
  proceduralIK: (landmarks: MediaPipeLandmarks) => void
  setAnimationController: (controller: AnimationController) => void
}

const useAnimationStore = create<AnimationControllerStore>((set, get) => ({
  currentClip: null,
  isPlaying: false,
  time: 0,
  weights: {},
  controller: null as AnimationController | null,

  play: (clipName: string, fadeTime: number = 0.5) => {
    const controller = get().controller
    if (controller) {
      controller.play(clipName, fadeTime)
    }
    set({ currentClip: clipName, isPlaying: true })
  },

  stop: (fadeTime: number = 0.5) => {
    const controller = get().controller
    if (controller) {
      controller.stop(fadeTime)
    }
    set({ isPlaying: false })
  },

  blend: (clipName: string, weight: number, fadeTime: number = 0.5) => {
    const controller = get().controller
    if (controller) {
      controller.blend(clipName, weight, fadeTime)
    }
    set((state) => ({
      weights: { ...state.weights, [clipName]: weight }
    }))
  },

  setTime: (time: number) => {
    const controller = get().controller
    if (controller) {
      controller.setTime(time)
    }
    set({ time })
  },

  proceduralIK: (landmarks: MediaPipeLandmarks) => {
    const controller = get().controller
    if (controller) {
      controller.proceduralIK(landmarks)
    }
  },

  setAnimationController: (controller: AnimationController) => {
    set({ controller })
  }
}))

export class AnimationController {
  private gltf: Group
  private mixer: AnimationMixer
  private actions: Map<string, any> = new Map()
  private clips: Map<string, AnimationClip> = new Map()
  private activeWeights: Map<string, number> = new Map()
  private landmarkMap: Map<string, number> = new Map()
  private bones: Map<string, Bone> = new Map()
  private skeleton: any
  private clock: any

  constructor(gltf: Group, clock?: any) {
    this.gltf = gltf
    this.mixer = new AnimationMixer(gltf)
    this.clock = clock || { getDelta: () => 0, elapsed: 0 }
    this.setupAnimations()
    this.indexBones()
    this.buildLandmarkMap()
  }

  private setupAnimations() {
    const Animations = (this.gltf as any).animations || []
    Animations.forEach((clip: AnimationClip) => {
      const action = this.mixer.clipAction(clip, this.gltf)
      this.actions.set(clip.name, action)
      this.clips.set(clip.name, clip)
    })
  }

  private indexBones() {
    this.gltf.traverse((obj: any) => {
      if (obj.isBone) {
        this.bones.set(obj.name, obj)
        if (!this.skeleton && obj.parent && (obj.parent as any).isSkeleton) {
          this.skeleton = obj.parent
        }
      }
    })
  }

  private buildLandmarkMap() {
    const landmarkNames = [
      'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
      'right_eye_inner', 'right_eye', 'right_eye_outer',
      'left_ear', 'right_ear',
      'mouth_left', 'mouth_right',
      'left_shoulder', 'right_shoulder',
      'left_elbow', 'right_elbow',
      'left_wrist', 'right_wrist',
      'left_pinky_finger', 'right_pinky_finger',
      'left_index_finger', 'right_index_finger',
      'left_thumb', 'right_thumb'
    ]
    
    landmarkNames.forEach((name, index) => {
      this.landmarkMap.set(name, index)
    })
  }

  play(clipName: string, fadeTime: number = 0.5): void {
    const action = this.actions.get(clipName)
    if (!action) return

    action.reset()
    action.setLoop(LoopRepeat)
    action.setEffectiveWeight(1)
    action.setEffectiveTimeScale(1)
    action.clampWhenFinished = true

    this.mixer.stopAllAction()
    action.fadeIn(fadeTime)
    action.play()
  }

  stop(fadeTime: number = 0.5): void {
    this.actions.forEach((action) => {
      action.fadeOut(fadeTime)
    })
    this.mixer.stopAllAction(fadeTime)
  }

  blend(clipName: string, weight: number, fadeTime: number = 0.5): void {
    const action = this.actions.get(clipName)
    if (!action) return

    action.setEffectiveWeight(weight)
    if (weight > 0 && !action.isPlaying) {
      action.play()
    }
    action.fadeTo(weight, fadeTime)
  }

  crossfade(fromClip: string, toClip: string, duration: number = 1.0): void {
    const fromAction = this.actions.get(fromClip)
    const toAction = this.actions.get(toClip)
    
    if (!fromAction || !toAction) return

    toAction.reset().play()
    fromAction.crossFadeTo(toAction, duration, true)
  }

  setTime(time: number): void {
    this.mixer.setTime(time)
  }

  getClipNames(): string[] {
    return Array.from(this.clips.keys())
  }

  getClip(clipName: string): AnimationClip | undefined {
    return this.clips.get(clipName)
  }

  hasClip(clipName: string): boolean {
    return this.clips.has(clipName)
  }

  update(deltaSeconds: number): void {
    if (this.mixer) {
      this.mixer.update(deltaSeconds)
    }
  }

  proceduralIK(landmarks: MediaPipeLandmarks): void {
    if (!this.skeleton) return

    const leftShoulder = landmarks.left_shoulder || landmarks['0']
    const leftElbow = landmarks.left_elbow || landmarks['1']
    const leftWrist = landmarks.left_wrist || landmarks['2']
    const rightShoulder = landmarks.right_shoulder || landmarks['3']
    const rightElbow = landmarks.right_elbow || landmarks['4']
    const rightWrist = landmarks.right_wrist || landmarks['5']

    this.applyIKChain('L', leftShoulder, leftElbow, leftWrist)
    this.applyIKChain('R', rightShoulder, rightElbow, rightWrist)

    this.applyFingerCurl(landmarks, 'L')
    this.applyFingerCurl(landmarks, 'R')
  }

  private applyIKChain(
    side: 'L' | 'R',
    shoulder: MediaPipeLandmark | undefined,
    elbow: MediaPipeLandmark | undefined,
    wrist: MediaPipeLandmark | undefined
  ): void {
    if (!shoulder || !elbow || !wrist) return

    const prefix = side === 'L' ? 'Left' : 'Right'
    
    const shoulderBone = this.bones.get(`${prefix}_Shoulder`)
    const elbowBone = this.bones.get(`${prefix}_Forearm`)
    const wristBone = this.bones.get(`${prefix}_Hand`)

    if (wristBone) {
      const target = new Vector3(
        wrist.x,
        -wrist.y,
        wrist.z
      )
      wristBone.position.copy(target)
    }

    if (elbowBone && wristBone) {
      const elbowTarget = new Vector3(
        elbow.x,
        -elbow.y,
        elbow.z
      )
      
      const dir = new Vector3().subVectors(elbowTarget, wristBone.position)
      dir.normalize()
      elbowBone.position.copy(elbowTarget)
    }

    if (shoulderBone && elbowBone) {
      const shoulderTarget = new Vector3(
        shoulder.x,
        -shoulder.y,
        shoulder.z
      )
      shoulderBone.position.copy(shoulderTarget)
    }
  }

  private applyFingerCurl(landmarks: MediaPipeLandmarks, side: 'L' | 'R'): void {
    const fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    
    fingers.forEach((finger, fingerIndex) => {
      const bone1 = this.bones.get(`${side}_${finger}_01`)
      const bone2 = this.bones.get(`${side}_${finger}_02`)
      const bone3 = this.bones.get(`${side}_${finger}_03`)

      const landmarks21 = Array.from({ length: 21 }, (_, i) => 
        landmarks[i.toString()] || landmarks[i]
      )

      const fingerStartIndex = fingerIndex * 4 + 1
      const tipIndex = fingerStartIndex + 3

      const tip = landmarks21[tipIndex]
      const base = landmarks21[fingerStartIndex]

      if (tip && base && bone1) {
        const dx = tip.x - base.x
        const dy = tip.y - base.y
        const dz = tip.z - base.z
        
        const bendAngle = Math.atan2(Math.sqrt(dx * dx + dz * dz), -dy)
        
        if (finger === 'thumb') {
          bone1.rotation.setFromVector3(new Vector3(0, 0, bendAngle * 0.5))
        } else {
          bone1.rotation.setFromVector3(new Vector3(bendAngle * 0.5, 0, 0))
        }
      }

      if (tip && base && bone2) {
        bone2.rotation.setFromVector3(new Vector3(0, 0, -0.2))
      }

      if (tip && base && bone3) {
        bone3.rotation.setFromVector3(new Vector3(0, 0, -0.1))
      }
    })
  }

  getActiveClipName(): string | null {
    for (const [name, action] of this.actions) {
      if (action.isPlaying) {
        return name
      }
    }
    return null
  }

  getWeight(clipName: string): number {
    return this.activeWeights.get(clipName) || 0
  }

  getClipWeight(clipName: string): number {
    const action = this.actions.get(clipName)
    return action ? action.getEffectiveWeight() : 0
  }

  dispose(): void {
    this.stop(0)
    this.mixer.stopAllAction()
    this.mixer = null as any
    this.actions.clear()
    this.clips.clear()
  }
}

const useAnimationController = (gltfPath: string) => {
  const gltf = useGLTF(gltfPath) as Group
  const gltfScene = (gltf as any).scene || gltf
  const { actions, mixer, clips, ref } = useAnimations((gltf as any).animations, gltfScene)
  
  const controllerRef = useRef<AnimationController | null>(null)
  const clockRef = useRef<any>({
    getDelta: () => 0,
    elapsed: 0
  })
  const [isLoaded, setIsLoaded] = useState(false)

  useEffect(() => {
    if (gltfScene) {
      controllerRef.current = new AnimationController(gltfScene, clockRef.current)
      useAnimationStore.getState().setAnimationController(controllerRef.current)
      setIsLoaded(true)
    }

    return () => {
      controllerRef.current?.dispose()
    }
  }, [gltfScene])

  useFrame((_, delta) => {
    if (controllerRef.current) {
      controllerRef.current.update(delta)
      clockRef.current.elapsed += delta
    }
  })

  return {
    isLoaded,
    controller: controllerRef.current,
    clips: clips,
    actions: actions,
    play: useCallback((clipName: string, fadeTime?: number) => 
      useAnimationStore.getState().play(clipName, fadeTime), []),
    stop: useCallback((fadeTime?: number) => 
      useAnimationStore.getState().stop(fadeTime), []),
    blend: useCallback((clipName: string, weight: number, fadeTime?: number) => 
      useAnimationStore.getState().blend(clipName, weight, fadeTime), []),
    crossfade: useCallback((from: string, to: string, duration?: number) => {
      if (controllerRef.current) {
        controllerRef.current.crossfade(from, to, duration)
      }
    }, []),
    setTime: useCallback((time: number) => 
      useAnimationStore.getState().setTime(time), []),
    proceduralIK: useCallback((landmarks: MediaPipeLandmarks) => 
      useAnimationStore.getState().proceduralIK(landmarks), []),
  }
}

export default useAnimationController
export { AnimationController, useAnimationStore, useAnimationController }
export type { MediaPipeLandmark, MediaPipeLandmarks }