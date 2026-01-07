import xml.etree.ElementTree as ET


def change_convention(urdf_path, output_path):
    # Parse the URDF file
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    old2new_link = get_new_link_names(tree)
    old2new_joint = get_new_joint_names(tree)

    for link in root.findall("link"):
        link_name = link.get("name")
        new_link_name = old2new_link[link_name]
        link.set("name", new_link_name)

    for joint in root.findall("joint"):
        joint_name = joint.get("name")
        new_joint_name = old2new_joint[joint_name]
        joint.set("name", new_joint_name)

        parent = joint.find("parent")
        parent_name = parent.get("link")
        new_parent_name = old2new_link[parent_name]
        parent.set("link", new_parent_name)

        child = joint.find("child")
        child_name = child.get("link")
        new_child_name = old2new_link[child_name]
        child.set("link", new_child_name)

    # Write the modified URDF back to a file
    tree.write(output_path, xml_declaration=True, encoding="utf-8")


def get_new_link_names(tree):
    root = tree.getroot()
    # Iterate over all links
    old2new_link = {}
    for link in root.findall("link"):
        link_name = link.get("name")
        old2new_link[link_name] = get_new_link_name(link_name)
    return old2new_link


def get_new_joint_names(tree):
    root = tree.getroot()
    # Iterate over all joints
    old2new_joint = {}
    for joint in root.findall("joint"):
        joint_name = joint.get("name")

        old2new_joint[joint_name] = get_new_joint_name(joint_name)
    return old2new_joint


def get_new_link_name(old_name):
    if old_name in ["Body", "world"]:
        return old_name
    if "Ankle" in old_name:
        return old_name
    new_qaudrant = get_quadrant(old_name)
    new_conv = get_new_link_convention(old_name)
    return new_conv + "_" + new_qaudrant


def get_new_joint_name(old_name):
    if "Ankle" in old_name:
        return old_name
    if "world" in old_name:
        return old_name
    new_qaudrant = get_quadrant(old_name)
    new_conv = get_new_joint_convention(old_name)
    return new_conv + "_" + new_qaudrant


def get_new_link_convention(old_name):
    old_conv = "_".join(old_name.split("_")[1:])
    match old_conv:
        case "motorHousing":
            return "MotorHousing"
        case "hip_outer":
            return "OuterThigh"
        case "shank_outer":
            return "OuterShank"
        case "hip_inner":
            return "InnerThigh"
        case "shank_inner":
            return "InnerShank"
        case "paw":
            return "Paw"
        case _:
            raise ValueError(f"Unknown link type: {old_conv}")


def get_new_joint_convention(old_name):
    if "_hipAduction_" in old_name:
        return "LateralMotor"
    if "_hip_outer" in old_name:
        return "OuterTransversalMotor"
    if "_hip_inner" in old_name:
        return "InnerTransversalMotor"
    if "_shank_outer_knee" in old_name:
        return "OuterKnee"
    if "_shank_inner_knee" in old_name:
        return "InnerKnee"
    if "_paw_joint" in old_name:
        return "PawJoint"

    raise ValueError(f"Unknown joint name: {old_name}")


def get_quadrant(old_name):
    old_qaudrant = old_name.split("_")[0]
    match old_qaudrant:
        case "FrontLeft":
            return "FL"
        case "FrontRight":
            return "FR"
        case "BackLeft":
            return "BL"
        case "BackRight":
            return "BR"
        case _:
            raise ValueError(f"Unknown quadrant: {old_qaudrant}")


if __name__ == "__main__":

    urdf_path = "olympus_old.urdf"
    output_path = "olympus.urdf"
    change_convention(urdf_path, output_path)

    print(f"Modified URDF saved to {output_path}")
