import bpy
from mathutils import Vector, Euler
import math

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)
    for data in [bpy.data.meshes, bpy.data.materials, bpy.data.armatures]:
        for block in list(data):
            data.remove(block)

def create_base_character():
    bpy.ops.mesh.primitive_cylinder_add(radius=0.15, depth=1.8, location=(0, 0, 0.9))
    body = bpy.context.active_object
    body.name = "SignAvatar_Base"
    bpy.ops.object.shade_smooth()
    return body

def build_custom_armature():
    arm_data = bpy.data.armatures.new("SignAvatar_Armature")
    arm_obj = bpy.data.objects.new("SignAvatar_Rig", arm_data)
    bpy.context.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')

    bones_config = [
        ('hip', (0, 0, 0.9), 'hip'),
        ('spine', (0, 0, 1.2), 'spine'),
        ('chest', (0, 0, 1.55), 'chest'),
        ('neck', (0, 0, 1.75), 'neck'),
        ('head', (0, 0, 1.88), 'head'),
        ('L_shoulder', (-0.25, 0, 1.55), 'L_shoulder'),
        ('L_upper_arm', (-0.45, 0, 1.55), 'L_upper_arm'),
        ('L_forearm', (-0.7, 0, 1.45), 'L_forearm'),
        ('L_hand', (-0.85, 0, 1.35), 'L_hand'),
        ('R_shoulder', (0.25, 0, 1.55), 'R_shoulder'),
        ('R_upper_arm', (0.45, 0, 1.55), 'R_upper_arm'),
        ('R_forearm', (0.7, 0, 1.45), 'R_forearm'),
        ('R_hand', (0.85, 0, 1.35), 'R_hand'),
        ('L_thigh', (-0.15, 0, 0.9), 'L_thigh'),
        ('L_shin', (-0.15, 0, 0.45), 'L_shin'),
        ('L_foot', (-0.15, 0, 0.05), 'L_foot'),
        ('R_thigh', (0.15, 0, 0.9), 'R_thigh'),
        ('R_shin', (0.15, 0, 0.45), 'R_shin'),
        ('R_foot', (0.15, 0, 0.05), 'R_foot'),
    ]

    edit_bones = {}
    for name, head, parent in bones_config:
        bone = arm_data.edit_bones.new(name)
        bone.head = Vector(head)
        bone.tail = Vector((head[0], head[1], head[2] + 0.15))
        edit_bones[name] = bone
        if parent and parent in edit_bones:
            bone.parent = edit_bones[parent]

    # 21 bones per hand
    for side in ['L', 'R']:
        hand_bone = edit_bones.get(f'{side}_hand')
        if hand_bone:
            hand_bone.tail = (hand_bone.head[0], hand_bone.head[1], hand_bone.head[2] - 0.05)

            # Thumb bones
            thumb_01 = arm_data.edit_bones.new(f'{side}_thumb_01')
            thumb_01.head = hand_bone.tail
            thumb_01.tail = (thumb_01.head.x + 0.03, thumb_01.head.y - 0.02, thumb_01.head.z - 0.05)
            thumb_01.parent = hand_bone

            thumb_02 = arm_data.edit_bones.new(f'{side}_thumb_02')
            thumb_02.head = thumb_01.tail
            thumb_02.tail = (thumb_02.head.x + 0.03, thumb_02.head.y - 0.02, thumb_02.head.z - 0.05)
            thumb_02.parent = thumb_01

            thumb_03 = arm_data.edit_bones.new(f'{side}_thumb_03')
            thumb_03.head = thumb_02.tail
            thumb_03.tail = (thumb_03.head.x + 0.03, thumb_03.head.y - 0.02, thumb_03.head.z - 0.05)
            thumb_03.parent = thumb_02

            thumb_end = arm_data.edit_bones.new(f'{side}_thumb_end')
            thumb_end.head = thumb_03.tail
            thumb_end.tail = (thumb_end.head.x + 0.02, thumb_end.head.y - 0.01, thumb_end.head.z - 0.03)
            thumb_end.parent = thumb_03

            # Index finger bones
            index_01 = arm_data.edit_bones.new(f'{side}_index_01')
            index_01.head = hand_bone.tail
            index_01.tail = (index_01.head.x + 0.04, index_01.head.y, index_01.head.z - 0.06)
            index_01.parent = hand_bone

            index_02 = arm_data.edit_bones.new(f'{side}_index_02')
            index_02.head = index_01.tail
            index_02.tail = (index_02.head.x + 0.03, index_02.head.y, index_02.head.z - 0.05)
            index_02.parent = index_01

            index_03 = arm_data.edit_bones.new(f'{side}_index_03')
            index_03.head = index_02.tail
            index_03.tail = (index_03.head.x + 0.03, index_03.head.y, index_03.head.z - 0.05)
            index_03.parent = index_02

            index_end = arm_data.edit_bones.new(f'{side}_index_end')
            index_end.head = index_03.tail
            index_end.tail = (index_end.head.x + 0.02, index_end.head.y, index_end.head.z - 0.03)
            index_end.parent = index_03

            # Middle finger bones
            middle_01 = arm_data.edit_bones.new(f'{side}_middle_01')
            middle_01.head = hand_bone.tail
            middle_01.tail = (middle_01.head.x + 0.04, middle_01.head.y + 0.01, middle_01.head.z - 0.06)
            middle_01.parent = hand_bone

            middle_02 = arm_data.edit_bones.new(f'{side}_middle_02')
            middle_02.head = middle_01.tail
            middle_02.tail = (middle_02.head.x + 0.03, middle_02.head.y + 0.01, middle_02.head.z - 0.05)
            middle_02.parent = middle_01

            middle_03 = arm_data.edit_bones.new(f'{side}_middle_03')
            middle_03.head = middle_02.tail
            middle_03.tail = (middle_03.head.x + 0.03, middle_03.head.y + 0.01, middle_03.head.z - 0.05)
            middle_03.parent = middle_02

            middle_end = arm_data.edit_bones.new(f'{side}_middle_end')
            middle_end.head = middle_03.tail
            middle_end.tail = (middle_end.head.x + 0.02, middle_end.head.y + 0.01, middle_end.head.z - 0.03)
            middle_end.parent = middle_03

            # Ring finger bones
            ring_01 = arm_data.edit_bones.new(f'{side}_ring_01')
            ring_01.head = hand_bone.tail
            ring_01.tail = (ring_01.head.x + 0.04, ring_01.head.y + 0.02, ring_01.head.z - 0.06)
            ring_01.parent = hand_bone

            ring_02 = arm_data.edit_bones.new(f'{side}_ring_02')
            ring_02.head = ring_01.tail
            ring_02.tail = (ring_02.head.x + 0.03, ring_02.head.y + 0.02, ring_02.head.z - 0.05)
            ring_02.parent = ring_01

            ring_03 = arm_data.edit_bones.new(f'{side}_ring_03')
            ring_03.head = ring_02.tail
            ring_03.tail = (ring_03.head.x + 0.03, ring_03.head.y + 0.02, ring_03.head.z - 0.05)
            ring_03.parent = ring_02

            ring_end = arm_data.edit_bones.new(f'{side}_ring_end')
            ring_end.head = ring_03.tail
            ring_end.tail = (ring_end.head.x + 0.02, ring_end.head.y + 0.02, ring_end.head.z - 0.03)
            ring_end.parent = ring_03

            # Pinky finger bones
            pinky_01 = arm_data.edit_bones.new(f'{side}_pinky_01')
            pinky_01.head = hand_bone.tail
            pinky_01.tail = (pinky_01.head.x + 0.03, pinky_01.head.y + 0.03, pinky_01.head.z - 0.05)
            pinky_01.parent = hand_bone

            pinky_02 = arm_data.edit_bones.new(f'{side}_pinky_02')
            pinky_02.head = pinky_01.tail
            pinky_02.tail = (pinky_02.head.x + 0.02, pinky_02.head.y + 0.03, pinky_02.head.z - 0.04)
            pinky_02.parent = pinky_01

            pinky_03 = arm_data.edit_bones.new(f'{side}_pinky_03')
            pinky_03.head = pinky_02.tail
            pinky_03.tail = (pinky_03.head.x + 0.02, pinky_03.head.y + 0.03, pinky_03.head.z - 0.04)
            pinky_03.parent = pinky_02

            pinky_end = arm_data.edit_bones.new(f'{side}_pinky_end')
            pinky_end.head = pinky_03.tail
            pinky_end.tail = (pinky_end.head.x + 0.015, pinky_end.head.y + 0.03, pinky_end.head.z - 0.025)
            pinky_end.parent = pinky_03

    bpy.ops.object.mode_set(mode='OBJECT')
    return arm_obj

def create_facs_blendshapes(obj):
    facs_shapes = {
        'brow_inner_up': 'AU1', 'brow_outer_up': 'AU2', 'brow_down': 'AU4',
        'upper_lid_raise': 'AU5', 'cheek_raise': 'AU6', 'lid_tighten': 'AU7',
        'nose_wrinkle': 'AU9', 'upper_lip_raise': 'AU10',
        'nasolabial_deepen': 'AU11', 'lip_corner_pull': 'AU12',
        'sharp_lip_pull': 'AU13', 'dimpler': 'AU14', 'lip_corner_depress': 'AU15',
        'chin_raise': 'AU17', 'lip_pucker': 'AU18', 'tongue_show': 'AU19',
        'lip_stretch': 'AU20', 'lip_tighten': 'AU23', 'lip_press': 'AU24',
        'lips_part': 'AU25', 'jaw_drop': 'AU26', 'mouth_stretch': 'AU27', 'lip_suck': 'AU28',
        'head_tilt_left': 'HEAD_TILT_L', 'head_tilt_right': 'HEAD_TILT_R',
        'head_nod': 'HEAD_NOD', 'head_shake': 'HEAD_SHAKE',
    }
    for shape_name in facs_shapes.keys():
        obj.shape_key_add(name=shape_name, from_mix=False)
    return list(facs_shapes.keys())

def main():
    clear_scene()
    body = create_base_character()
    rig = build_custom_armature()

    # Set armature modifier on body
    mod = body.modifiers.new(name="Armature", type='ARMATURE')
    mod.object = rig

    # Create FACS blendshapes
    facs_shapes = create_facs_blendshapes(body)

    # Parent body to rig for animation
    body.parent = rig
    bpy.ops.object.select_all(action='DESELECT')
    body.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')

    # Export GLTF
    export_path = "C:/Users/Sanju/Projects/ML Flow/.kilo/worktrees/avatar-pipeline/frontend/public/avatars/sign_avatar.gltf"
    bpy.ops.export_scene.gltf(
        filepath=export_path,
        export_format='GLTF_SEPARATE',
        export_morph=True,
        export_morph_normal=True,
        export_skins=True,
        export_animations=True,
        export_frame_range=True,
        export_current_frame=False,
        export_def_bones=True,
    )

    blend_path = "C:/Users/Sanju/Projects/ML Flow/.kilo/worktrees/avatar-pipeline/blender/character/sign_avatar.blend"
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)

    print("Character setup complete!")
    print(f"Rig bones: {len(rig.data.bones)} (body + 21*2 hands + face)")
    print(f"FACS shapes: {len(facs_shapes)}")
    print(f"Exported GLTF: {export_path}")
    print(f"Saved blend: {blend_path}")

if __name__ == "__main__":
    main()
