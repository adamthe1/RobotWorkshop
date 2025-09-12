#!/usr/bin/env python3
"""
MuJoCo Scene Generator for Panda Robot Bar Scene
Allows easy Y-offset adjustment and multiple robot creation with incrementing prefixes.
All objects (except floor) duplicate with each robot using the same spacing.
"""
import os
from dotenv import load_dotenv

load_dotenv()


def generate_mujoco_xml(y_offset=0.0, num_robots=1, robot_spacing=2.0, 
                        use_scene=1, add_franka_pedestal=True) -> str:
    """
    Generate MuJoCo XML for panda robot bar scene.
    
    Args:
        y_offset (float): Y-axis offset to apply to all objects
        num_robots (int): Number of robots to create
        robot_spacing (float): Spacing between robots in Y direction
    
    Returns:
        str: Complete MuJoCo XML string
    """

    num_franka = int(os.getenv("FRANKA_PANDA_COUNT", 1))
    num_so101 = int(os.getenv("SO101_COUNT", 1))
    robot_dict = {**{f'panda{i}': "FrankaPanda" for i in range(num_franka)},
                  **{f'so101_{i}': "SO101" for i in range(num_so101)}}

    # Helper function to apply Y offset to position
    def pos_with_offset(x, y, z, robot_index=0):
        total_y = y + y_offset + (robot_index * robot_spacing)
        return f"{x} {total_y} {z}"
    
    # Start building XML
    xml_content = f'''<mujoco model="panda scene (bar table)">

  <option integrator="implicitfast"/>
  <compiler angle="radian" meshdir="assets" autolimits="true" assetdir="meshes"/>

  <statistic center="0.3 0 0.4" extent="1"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>

    <mesh name="wine_bottle_mesh" file="WineBottle.obj"/>
    <mesh name="wine_glass_mesh" file="wine_glass.obj"/>

    <material name="bar_mat" rgba="0.8 0.8 0.8 1" texture="marble_tex"
            texuniform="true"
            texrepeat="1 1"
            reflectance="0.2" shininess="0.5" specular="1"/>

       
    <!-- Red liquid -->
    <material name="liquid_red" rgba="0.6 0.0 0.0 0.9" reflectance="0.2" shininess="0.3" specular="0.4"/>

    <material name="glass_mat" rgba="1 0.9 0.9 0.5" reflectance="0.5" shininess="0.9" specular="1"/>
    <material name="glass_green" rgba="0.2 0.8 0.2 0.6" reflectance="0.5" shininess="0.9" specular="1"/>
    <material name="glass_yellow" rgba="0.9 0.9 0.2 0.6" reflectance="0.5" shininess="0.9" specular="1"/>
    <!-- Beer liquid + coaster materials -->
    <material name="beer_mat" rgba="0.95 0.75 0.2 0.85" reflectance="0.2" shininess="0.2" specular="0.3"/>
    <material name="coaster_mat" rgba="0.8 0.2 0.2 1" reflectance="0.05" shininess="0.1" specular="0.1"/>

    <material name="wine_mat"
              rgba="0.6 0.0 0.0 0.8" 
              reflectance="0.2"
              shininess="0.3"
              specular="0.4"/>

    <texture name="marble_tex"
               type="2d"
               file="grey.png"
               width="2000"
               height="1000"/>

    <texture name="wood_tex"
               type="2d"
               file="wood3.png"
               width="20"
               height="20"/>

    <!-- 2. Create a material that uses that texture -->
    <material name="wood_floor_mat"
              texture="wood_tex"
              texuniform="true"
              texrepeat="2.5 2.5"
              reflectance="1"
              shininess="0"
              specular="1"/> 

    <model name="panda" file="panda_for_duplicating.xml"/>
    <model name="so101" file="so101_for_duplicating.xml"/>
  </asset>

  <worldbody>'''

    # Add shared floor (only object that doesn't duplicate)
    xml_content += f'''
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <body name="wood_floor" pos="0 0 0">
      <geom type="box" size="200 200 0.01"
            material="wood_floor_mat"
            contype="0" conaffinity="0"/>
    </body>'''

   

    # Add complete scene for each robot
    for j, robot_id in enumerate(list(robot_dict.keys())):
        i = j + 1
        prefix = f"{robot_id}_" if i > 0 else ""

        if robot_dict[robot_id] == "FrankaPanda":
            if add_franka_pedestal:
                xml_content += f'''
            <!-- ==================== ROBOT {i} SCENE ==================== -->
    <!-- Robot {i} -->
    <body name="robot{i}" pos="{pos_with_offset(0.05 ,0 ,0, j)}">
      <attach model="panda" body="link0" prefix="{prefix}"/>
    </body>'''
            else:
                xml_content += f'''
              <!-- Robot {i} -->
    <body name="robot{i}" pos="{pos_with_offset(0.05 ,0 ,0, j)}">
      <attach model="panda" body="link0" prefix="{prefix}"/>
    </body>'''

        elif robot_dict[robot_id] == "SO101":
            xml_content += f'''
            <!-- ==================== ROBOT {i} SCENE ==================== -->

    <!-- Robot {i} -->
    <body name="robot{i}" pos="{pos_with_offset(0.05 ,-0.2 ,0, j)}">
      <attach model="so101" body="base" prefix="{prefix}"/>
    </body>'''
            
      # Robot Bar Scene (common for both robots)
        if use_scene == 1:
            xml_content += get_scene1(i, j, pos_with_offset)
        else:
            xml_content += get_scene2(i, j, pos_with_offset)

    xml_content += '''

  </worldbody>
</mujoco>'''

    return xml_content, robot_dict


def get_scene1(i, j, pos_with_offset):
     # Just before the per-robot loop, add handy heights/positions for table/pedestals/coasters:
    table_top_z = 0.55          # main table top raised by +0.1 (now 0.75)
    initial_z = 0.3
    ped_h = 0.0                     # small pedestal halfheight
    ped_top_z = table_top_z + ped_h   # pedestal top z (center of pedestal body)
    coaster_h =  0.01                 # thin coaster halfheight
    # Beer glass: add 1mm clearance so it settles without a jump
    beer_glass_pos = (0.13, -0.5, initial_z+table_top_z+0.1)
    green_bottle_pos = (0.05, 0.5, initial_z+table_top_z+0.1)
    yellow_bottle_pos = (0.2, 0.5, initial_z+table_top_z+0.1)

    return f'''

    <!-- Bar wall for robot {i} -->
    <body name="bar_wall{i}" pos="{pos_with_offset(-1.0, 0.0, initial_z+0.1, j)}">
      <geom type="box" size="0.1 1.0 0.8" material="bar_mat" density="2000" contype="1" conaffinity="1"/>
    </body>

    <!-- Table for robot {i} (raised +0.1) -->
    <body name="table{i}" pos="{pos_with_offset(0.8, 0.0, initial_z, j)}">
      <geom type="box" size="0.3 0.9 {table_top_z}" material="bar_mat" contype="1" conaffinity="1" density="2000"/>
    </body>

    <body name="table_2_{i}" pos="{pos_with_offset(0.2, 0.6, initial_z, j)}">
      <geom type="box" size="0.3 0.2 {ped_top_z}" material="bar_mat" contype="1" conaffinity="1" density="2000"/>
    </body>

    <body name="table_3_{i}" pos="{pos_with_offset(0.2, -0.6, initial_z, j)}">
      <geom type="box" size="0.3 0.2 {ped_top_z}" material="bar_mat" contype="1" conaffinity="1" density="2000"/>
    </body>


    <!-- Small center coaster on the table (reference for cups) -->
    <body name="center_coaster{i}" pos="{pos_with_offset(0.63, 0.0, initial_z+ table_top_z  , j)}">
      <geom type="cylinder" size="0.07 {coaster_h}" material="coaster_mat" contype="0" conaffinity="0"/>
    </body>

    <body name="beer_glass{i}" pos="{pos_with_offset(beer_glass_pos[0], beer_glass_pos[1], beer_glass_pos[2], j)}">
      <joint name="beer_glass_free{i}" type="free" />
      <!-- Outer glass (square / box shape) -->
      <geom type="box"
        size="0.03 0.03 0.10"
        material="glass_mat"
        mass="0.35"
        contype="65535" conaffinity="65535" condim="6"
        friction="5.8 0.25 0.05"
        solimp="0.95 0.995 0.0005"
        solref="0.004 1"/>


      <geom name="beer_glass_base{i}"
        type="box"
        size="0.048 0.048 0.005"    
        pos="0 0 -0.1"          
        material="glass_mat"
        mass="0.06"               
        contype="65535"
        conaffinity="65535"
        condim="6"
        friction="5.5 0.22 0.05"  
        solimp="0.95 0.995 0.0005"
        solref="0.004 1" />
    </body>
    
    <!-- Green square bottle (right pedestal, right coaster) -->
    <body name="green_bottle_body{i}" pos="{pos_with_offset(green_bottle_pos[0], green_bottle_pos[1], green_bottle_pos[2], j)}">
      <joint name="green_bottle_free{i}" type="free"/>
      <geom name="green_bottle_body{i}" type="box" size="0.035 0.035 0.14" material="glass_green" mass="0.3"
            contype="1" conaffinity="1" condim="6" friction="5.408 0.2366 0.04225" solimp="0.95 0.995 0.0005" solref="0.004 1" />
      <geom name="green_bottle_base{i}" type="box" size="0.048 0.048 0.005" pos="0 0 -0.135" material="glass_green"
            mass="0.02" contype="1" conaffinity="1" condim="6" friction="5.408 0.2366 0.04225" solimp="0.95 0.995 0.0005" solref="0.004 1" />
      <geom name="green_bottle_neck{i}" type="box" size="0.012 0.012 0.06" pos="0 0 0.20" material="glass_green"
            mass="0.09" contype="1" conaffinity="1" condim="6" friction="5.408 0.2366 0.04225" solimp="0.95 0.995 0.0005" solref="0.004 1" />
    </body>

    <!-- Yellow square bottle (right pedestal, left coaster) -->
    <body name="yellow_bottle_body{i}" pos="{pos_with_offset(yellow_bottle_pos[0], yellow_bottle_pos[1], yellow_bottle_pos[2], j)}">
      <joint name="yellow_bottle_free{i}" type="free"/>
      <geom name="yellow_bottle_body{i}" type="box" size="0.035 0.035 0.14" material="glass_yellow" mass="0.3"
            contype="1" conaffinity="1" condim="6" friction="5.408 0.2366 0.04225" solimp="0.95 0.995 0.0005" solref="0.004 1" />
      <geom name="yellow_bottle_base{i}" type="box" size="0.048 0.048 0.005" pos="0 0 -0.135" material="glass_yellow"
            mass="0.02" contype="1" conaffinity="1" condim="6" friction="5.408 0.2366 0.04225" solimp="0.95 0.995 0.0005" solref="0.004 1" />
      <geom name="yellow_bottle_neck{i}" type="box" size="0.012 0.012 0.06" pos="0 0 0.20" material="glass_yellow"
            mass="0.09" contype="1" conaffinity="1" condim="6" friction="5.408 0.2366 0.04225" solimp="0.95 0.995 0.0005" solref="0.004 1" />
    </body>
    '''

def get_scene2(i, j, pos_with_offset):
    return f'''<!-- Bar wall for robot {i} -->
<body name="bar_wall{i}" pos="{pos_with_offset(-1.0, 0.0, 0.4, j)}">
  <geom type="box"
        size="0.1 1.0 0.8"
        material="bar_mat"
        density="2000"
        contype="1" conaffinity="1"/>
</body>

<!-- Table for robot {i} -->
<body name="table{i}" pos="{pos_with_offset(0.8, 0.0, 0.3, j)}">
  <geom type="box" size="0.3 0.9 0.35" material="bar_mat" contype="1" conaffinity="1" density="2000"/>
</body>

<!-- Wine bottle for robot {i} -->
<body name="wine_bottle_body{i}" pos="{pos_with_offset(0.6, -0.25, 0.80, j)}">
  <joint name="wine_bottle_free{i}" type="free"/>
  <geom type="mesh"
        mesh="wine_bottle_mesh"
        material="glass_mat"
        contype="1"
        density="700"
        friction="3.0 0.08 0.008"
        conaffinity="1"/>
  <geom type="cylinder" pos="0 0 0.085" size="0.03 0.08" material="liquid_red" density="500" contype="1" conaffinity="1" friction="2.0 0.08 0.008"/>
</body>


<!-- Green square bottle for robot {i} -->
<body name="green_bottle_body{i}" pos="{pos_with_offset(0.6, 0.05, 0.79, j)}">
  <joint name="green_bottle_free{i}" type="free"/>
<geom name="green_bottle_body{i}" type="box" size="0.035 0.035 0.14" material="glass_green" mass="0.5"
        contype="1" conaffinity="1" condim="6" friction="5.408 0.2366 0.04225"
        solimp="0.95 0.995 0.0005" solref="0.004 1" />
  <!-- wider base plate to improve stability when placing the bottle down -->
  <geom name="green_bottle_base{i}" type="box" size="0.048 0.048 0.005" pos="0 0 -0.135" material="glass_green"
        mass="0.02" contype="1" conaffinity="1" condim="6" friction="5.408 0.2366 0.04225"
        solimp="0.95 0.995 0.0005" solref="0.004 1" />
  <!-- square bottle neck (box) on top of body: halfheight 0.06 placed 0.20 above body center (0.14 + 0.06) -->
  <geom name="green_bottle_neck{i}" type="box" size="0.012 0.012 0.06" pos="0 0 0.20" material="glass_green"
        mass="0.09" contype="1" conaffinity="1" condim="6" friction="5.408 0.2366 0.04225"
        solimp="0.95 0.995 0.0005" solref="0.004 1" />
</body>

<!-- Yellow square bottle for robot {i} -->
<body name="yellow_bottle_body{i}" pos="{pos_with_offset(0.6, -0.1, 0.79, j)}">
  <joint name="yellow_bottle_free{i}" type="free"/>
        <geom name="yellow_bottle_body{i}" type="box" size="0.035 0.035 0.14" material="glass_yellow" mass="0.5"
        contype="1" conaffinity="1" condim="6" friction="5.408 0.2366 0.04225"
        solimp="0.95 0.995 0.0005" solref="0.004 1" />
  <!-- wider base plate to improve stability when placing the bottle down -->
  <geom name="yellow_bottle_base{i}" type="box" size="0.048 0.048 0.005" pos="0 0 -0.135" material="glass_yellow"
        mass="0.02" contype="1" conaffinity="1" condim="6" friction="5.408 0.2366 0.04225"
        solimp="0.95 0.995 0.0005" solref="0.004 1" />
  <!-- square bottle neck (box) on top of body: halfheight 0.06 placed 0.20 above body center (0.14 + 0.06) -->
  <geom name="yellow_bottle_neck{i}" type="box" size="0.012 0.012 0.06" pos="0 0 0.20" material="glass_yellow"
        mass="0.09" contype="1" conaffinity="1" condim="6" friction="5.408 0.2366 0.04225"
        solimp="0.95 0.995 0.0005" solref="0.004 1" />
</body>
'''

def save_xml_file(filename, y_offset=0.0, num_robots=1, robot_spacing=2.0):
    """
    Generate and save MuJoCo XML file.
    
    Args:
        filename (str): Output filename
        y_offset (float): Y-axis offset to apply to all objects
        num_robots (int): Number of robots to create
        robot_spacing (float): Spacing between robots in Y direction
    """
    xml_content, robot_dict = generate_mujoco_xml(y_offset, num_robots, robot_spacing)
    
    with open(filename, 'w') as f:
        f.write(xml_content)
    
    return filename, robot_dict



# Example usage and main function
if __name__ == "__main__":

    # Example 4: Five robots with tight spacing
    pwd = os.getenv("MAIN_DIRECTORY", "/root/RobotWorkshop")
    save_xml_file(f"{pwd}/xml_robots/panda_scene_one_robot_franka.xml", y_offset=0.0, robot_spacing=2.5)

    # Show layout summary for the multi-robot example
    
