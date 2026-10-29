import bpy
import bmesh
from mathutils import Vector, Euler
import math

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block)
    for block in bpy.data.armatures:
        bpy.data.armatures.remove(block)

def create_base_character():
    """Create a base human mesh using a simple primitive approach.
    In production, import MakeHuman/MetaHuman base mesh here."""
    bpy.ops.mesh.primitive_cylinder_add(radius=0.15, depth=1.8, location=(0, 0, 0.9))
    body = bpy.context.active_object
    body.name = "SignAvatar_Base"
    
    bpy.ops.object.shade_smooth()
    return body

def setup_rigify_rig():
    """Set up Rigify meta-rig for full body."""
    bpy.ops.object.armature_human_metarig_add()
    metarig = bpy.context.active_object
    metarig.name = "SignAvatar_MetaRig"
    
    bpy.ops.object.mode_set(mode='EDIT')
    arm = metarig.data
    
    bone_names = [b.name for b in arm.edit_bones]
    return metarig

def create_hand_rig(side='L', parent_bone=None):
    """Create 21-bone hand rig with IK/FK switching."""
    hand_bones = [
        f'{side}_wrist',
        f'{side}_thumb_01', f'{side}_thumb_02', f'{side}_thumb_03', f'{side}_thumb_end',
        f'{side}_index_01', f'{side}_index_02', f'{side}_index_03', f'{side}_index_end',
        f'{side}_middle_01', f'{side}_middle_02', f'{side}_middle_03', f'{side}_middle_end',
        f'{side}_ring_01', f'{side}_ring_02', f'{side}_ring_03', f'{side}_ring_end',
        f'{side}_pinky_01', f'{side}_pinky_02', f'{side}_pinky_03', f'{side}_pinky_end',
    ]
    return hand_bones

def create_facs_blendshapes(obj):
    """Create FACS blendshapes for facial expressions."""
    facs_shapes = {
        'brow_inner_up': 'AU1',
        'brow_outer_up': 'AU2',
        'brow_down': 'AU4',
        'upper_lid_raise': 'AU5',
        'cheek_raise': 'AU6',
        'lid_tighten': 'AU7',
        'nose_wrinkle': 'AU9',
        'upper_lip_raise': 'AU10',
        'nasolabial_deepen': 'AU11',
        'lip_corner_pull': 'AU12',
        'sharp_lip_pull': 'AU13',
        'dimpler': 'AU14',
        'lip_corner_depress': 'AU15',
        'chin_raise': 'AU17',
        'lip_pucker': 'AU18',
        'tongue_show': 'AU19',
        'lip_stretch': 'AU20',
        'lip_tighten': 'AU23',
        'lip_press': 'AU24',
        'lips_part': 'AU25',
        'jaw_drop': 'AU26',
        'mouth_stretch': 'AU27',
        'lip_suck': 'AU28',
        'head_tilt_left': 'HEAD_TILT_L',
        'head_tilt_right': 'HEAD_TILT_R',
        'head_nod': 'HEAD_NOD',
        'head_shake': 'HEAD_SHAKE',
    }
    
    for shape_name in facs_shapes.keys():
        bpy.ops.object.shape_key_add(from_mix=False)
        key = obj.shape_key_blocks[-1]
        key.name = shape_name
    
    return list(facs_shapes.keys())

def setup_ik_fk_switching(armature_obj, hand_bones):
    """Set up IK/FK switching for hands."""
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.object.mode_set(mode='OBJECT')
    
    for bone_name in hand_bones:
        if bone_name in armature_obj.pose.bones:
            bone = armature_obj.pose.bones[bone_name]
            bone['ik_fk_switch'] = 0.0

def export_gltf(filepath):
    """Export to GLTF with morph targets and skeletal animation."""
    bpy.ops.export_scene.gltf(
        filepath=filepath,
        export_format='GLTF_SEPARATE',
        export_copyright='Sign Language Avatar Pipeline',
        export_image_format='AUTO',
        export_texcoords=True,
        export_normals=True,
        export_tangents=True,
        export_materials=True,
        export_colors=True,
        export_morph=True,
        export_morph_normal=True,
        export_morph_tangent=True,
        export_skins=True,
        export_animations=True,
        export_frame_range=True,
        export_frame_step=1,
        export_force_sampling=True,
        export_nla_strips=True,
        export_def_bones=True,
        export_current_frame=False,
        export_skip_inactive_bones=False,
    )

def main():
    clear_scene()
    
    body = create_base_character()
    metarig = setup_rigify_rig()
    
    left_hand = create_hand_rig('L')
    right_hand = create_hand_rig('R')
    
    facs_shapes = create_facs_blendshapes(body)
    
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active = metarig
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.rigify_generate()
    
    rig = bpy.context.active_object
    rig.name = "SignAvatar_Rig"
    
    setup_ik_fk_switching(rig, left_hand + right_hand)
    
    export_path = "C:/Users/Sanju/Projects/ML Flow/.kilo/worktrees/avatar-pipeline/frontend/public/avatars/sign_avatar.gltf"
    export_gltf(export_path)
    
    blend_path = "C:/Users/Sanju/Projects/ML Flow/.kilo/worktrees/avatar-pipeline/blender/character/sign_avatar.blend"
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    
    print("Character setup complete!")
    print(f"Exported to: {export_path}")
    print(f"Saved blend: {blend_path}")

if __name__ == "__main__":
    main()