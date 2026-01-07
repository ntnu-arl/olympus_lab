import xml.etree.ElementTree as ET

def set_diagonal_inertia_and_mass(urdf_path, output_path):
    # Parse the URDF file
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Iterate over all links
    for link in root.findall('link'):
        # Find the inertia element
        inertia = link.find('./inertial/inertia')
        if inertia is not None:
            # Set all off-diagonal inertia elements to 0
            inertia.set('ixy', '0.0')
            inertia.set('ixz', '0.0')
            inertia.set('iyz', '0.0')
            inertia.set('ixx', '1.0')
            inertia.set('iyy', '1.0')
            inertia.set('izz', '1.0')

        # Find the mass element and set it to 1
        mass = link.find('./inertial/mass')
        if mass is not None:
            mass.set('value', '1.0')

    # Write the modified URDF back to a file
    tree.write(output_path, xml_declaration=True, encoding='utf-8')

# Example usage
urdf_path = 'olympus.urdf'
output_path = 'olympus_kin.urdf'
set_diagonal_inertia_and_mass(urdf_path, output_path)

print(f"Modified URDF saved to {output_path}")