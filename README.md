# Sorcar

Procedural modeling in Blender using Node Editor

![sc_cover](https://github.com/aachman98/sc-img-data/raw/master/sc_cover.png "Sorcar v3")
<!-- [![sc_](http://img.youtube.com/vi/VIDEO_ID/0.jpg)](http://www.youtube.com/watch?v=VIDEO_ID "Sorcar v3") -->
</br>BlenderArtist Thread: <https://blenderartists.org/t/sorcar-procedural-modeling-in-blender-using-node-editor/1156769>
</br>Intro & Tutorials: <https://www.youtube.com/playlist?list=PLZiIC3gdS_O7nCm1-xpWbZmTQWeL5c6KV>
</br>Trello (Project Tracker): <https://trello.com/b/aKIFRoTh/sorcar>
</br>Documentation: https://github.com/aachman98/Sorcar/wiki

## About

Sorcar is a **procedural modeling node-based system** which utilises Blender and its Python API to create a visual programming environment for artists and developers. Heavily inspired by Side-FX Houdini, it presents a node editor with a variety of **modular nodes** to make the modelling workflow easier and fast. Most of the nodes are blender internal operations (bpy.ops.mesh/object) which also makes it easier for frequent blender users to manipulate geometry. It helps the users to quickly create 3D models and **control node parameters** to generate limitless variations in a **non-destructive** manner. It also provides the users to view and edit mesh on any stage of the node network independently, with **realtime updates**.

## Release & Instructions

[Latest Release (v3.2.1)](https://github.com/aachman98/Sorcar/releases/latest)
</br>*Requirement: Blender 2.81 or later*

1. Download the zip file and install it as a Blender addon (Edit -> Preferences... -> Add-ons -> Install...)
2. Open Sorcar Node Editor (**Do not** remove the 3D viewport as it is required by some operations like extrude, transform, ...)
3. Click on the + button to create a new tree
4. Press Shift+A to open the nodes menu. Alternatively, navigate through tabs on the Right panel in the node editor
5. Select the desired node and press "Set Preview"

*Open blender using a command prompt to view logs and errors, if encountered.*

## Features

| | |
| --- | --- |
| ![sc_visual_programming](https://github.com/aachman98/sc-img-data/raw/master/sc_visual_programming.gif "Visual Programming") | <p style="text-align: left; padding-left: 16px;"><strong>VISUAL PROGRAMMING</strong><br /><em>Don't like programming?</em></p> <hr style="padding-left: 16px;" /> <p style="text-align: left; padding-left: 16px;">Construct geometries using custom algorithms, maths or generate patterns without writing a single line of code!</p> |
| <p style="text-align: right; padding-right: 16px;"><strong>NON-DESTRUCTIVE WORKFLOW</strong><br /><em>Want to change cylinder vertices after bevel?</em></p> <hr style="padding-right: 16px;" /> <p style="text-align: right; padding-right: 16px;">Edit node parameters at any point without the fear of losing mesh data. Also apply same procedural operations to different objects easily.</p> | ![sc_non_destructive](https://github.com/aachman98/sc-img-data/raw/master/sc_non_destructive.gif "Non-Destructive Workflow") |
| ![sc_realtime_updates](https://github.com/aachman98/sc-img-data/raw/master/sc_realtime_updates.gif "Realtime Updates") | <p style="text-align: left; padding-left: 16px;"><strong>REAL-TIME UPDATES</strong><br /><em>Quick as the wind...</em></p> <hr style="padding-left: 16px;" /> <p style="text-align: left; padding-left: 16px;">Drive a parameter using current frame value (or manually change it) and see the mesh update in viewport.</p> |
| <p style="text-align: right; padding-right: 16px;"><strong>ITERATE & RANDOMIZE</strong><br /><em>Need multiple extrusions of random amount?</em></p> <hr style="padding-right: 16px;" /> <p style="text-align: right; padding-right: 16px;">Generate variations in mesh by using seed-controlled pseudorandom numbers. Use loops to handle repeatitive operations with same level of randomness.</p> | ![sc_iterate_randomize](https://github.com/aachman98/sc-img-data/raw/master/sc_iterate_randomize.gif "Iterate & Randomize") |
| ![sc_automation](https://github.com/aachman98/sc-img-data/raw/master/sc_automation.gif "Automation") | <p style="text-align: left; padding-left: 16px;"><strong>AUTOMATION</strong><br /><em>Modify, Save, Repeat...</em></p> <hr style="padding-left: 16px;" /> <p style="text-align: left; padding-left: 16px;">Use frame number to drive seed value and batch export the meshes in different files.</p> |
| <p style="text-align: right; padding-right: 16px;"><strong>250+ NODES</strong><br /><em>At your service!</em></p> <hr style="padding-right: 16px;" /> <p style="text-align: right; padding-right: 16px;">A growing list of functions available as nodes (operators & scene settings) including custom inputs, curve/mesh conversion, selection & transform tools, modifiers and component level operators.</p> | ![sc_nodes](https://github.com/aachman98/sc-img-data/raw/master/sc_nodes.gif "250+ Nodes") |

- Node-Groups to collapse big node networks into a single node with custom inputs & outputs
- Simplified node sockets with internal data conversion for the convenience of users.
- Colour-coded nodes (preview, error, invalid inputs etc.) for easier debugging.
- Multi-level heirarchy & auto-registration of classes for easy development of custom nodes in any category (existing or new).

and more...!

## Nodes

| | |
| --- | --- |
| ![sc_inputs](https://github.com/aachman98/sc-img-data/raw/master/sc_inputs.png "Inputs") | **Inputs** </br> Primitive Meshes (Cube, Cylinder, Sphere, ...), Empty Object, Import FBX, Custom Object from the scene, Create Object using arrays of Vertices, Edges & Faces  |
| ![sc_curves](https://github.com/aachman98/sc-img-data/raw/master/sc_curves.png "Curves") | **Curves** </br> Custom Curve from the scene, Text, Import SVG, Geometry/Shape/Spline Properties, Mesh-Curve or Curve-Mesh Conversion |
| | |
| ![sc_transform](https://github.com/aachman98/sc-img-data/raw/master/sc_transform.png "Transform") | **Transform** </br> Set/Add/Randomize transform (Edit/Object mode), Apply/Copy transform in world/local axes, Create custom orientation|
| ![sc_selection](https://github.com/aachman98/sc-img-data/raw/master/sc_selection.png "Selection") | **Selection** </br> Manual, invert/toggle, loops, random, similar components or by their property (location, index, normal, material, ...) |
| ![sc_deletion](https://github.com/aachman98/sc-img-data/raw/master/sc_deletion.png "Deletion") | **Deletion** </br> Delete/Dissolve selected components (or loops) |
| | |
| ![sc_component_operators](https://github.com/aachman98/sc-img-data/raw/master/sc_component_operators.png "Component Operators") | **Component Operators** </br> Bevel, Decimate, Extrude, Fill, Inset, Loop Cut, Merge, Offset Loop, Poke, Screw, Spin, Subdivide, UV Map, Assign material or vertex groups |
| ![sc_object_operators](https://github.com/aachman98/sc-img-data/raw/master/sc_object_operators.png "Object Operators") | **Object Operators** </br> Duplicate, Raycast/Overlap, Merge, Scatter, Set/Clear Parent, Voxel/QuadriFlow Remesh, Shading, Viewport Draw Mode |
| ![sc_modifiers](https://github.com/aachman98/sc-img-data/raw/master/sc_modifiers.png "Modifiers") | **Modifiers** </br> Array, Bevel, Boolean, Build, Cast, Curve, Decimate, Remesh, Shrinkwrap, Skin, Solidify, Subsurf, Wave, Weighted Normal, Wireframe |
| | |
| ![sc_constants](https://github.com/aachman98/sc-img-data/raw/master/sc_constants.png "Constants") | **Constants** </br> Number (Float/Int/Angle/Random), Bool, Vector, String, Selection Type (Face/Vert/Edge) |
| ![sc_arrays](https://github.com/aachman98/sc-img-data/raw/master/sc_arrays.png "Arrays") | **Arrays** </br> Create/Append Arrays, Add/Remove elements, Reverse, Search, Clear, Count, Get |
| | |
| ![sc_noise](https://github.com/aachman98/sc-img-data/raw/master/sc_noise.png "Noise") | **Noise** </br> Cell (Vector/Float), Fractal, Multi-Fractal, Hetero-Terrain, Ridged, Turbulence (Vector/Float), Variable Lacunarity, Voronoi |
| | |
| ![sc_utilities](https://github.com/aachman98/sc-img-data/raw/master/sc_utilities.png "Utilities") | **Utilities** </br> String/Bool/Vector ops, Maths, Clamp, Map, Trigonometry, Get/Set Variables, Node-Groups, Scene/Component/Object Info, Custom Python Script |
| ![sc_flow_control](https://github.com/aachman98/sc-img-data/raw/master/sc_flow_control.png "Flow Control") | **Flow Control** </br> For loop, For-Each loop (Array/Components), If-Else Branch |
| ![sc_settings](https://github.com/aachman98/sc-img-data/raw/master/sc_settings.png "Settings") | **Settings** </br> Cursor Transform, Edit Mode, Pivot Point, Snapping, Proportional Editing, Transform Orientation |

## Upcoming Feature

1. Export nodes as python code
2. Send to UE4/Unity (live-link)
3. Point scatter & voronoi fracture nodes
4. Debugging tools: Watch/track values of node parameters

## Future

1. Complete integration with Blender's internal dependency graph
2. Node-Viewport link: Create nodes automatically in editor based on actions in 3D viewport
3. Landscape (noise/masking/erosion) and Foliage (grass/bush/trees) nodes

## Showcase

![sc_logo](https://github.com/aachman98/sc-img-data/raw/master/sc_logo.png "Sorcar")
![sc_showcase](https://github.com/aachman98/sc-img-data/raw/master/sc_showcase.png "Made in Sorcar")

## Changelog

#### [Unreleased]

- Added "Debug" section under "Sorcar" tab in N-panel to show node properties (Inputs, Outputs, Internal, Inherited)
- Added "debug.py" to provide support for text-block/console log outputs of different levels
- Added support for timestamp and duration for log outputs
- Added "Flush Stacktrace" operator in "Debug" panel to force output logs for last execution
- Added Enum property to set default log level in Preferences
- Added support for custom menu & sub-menu layout for categories and nodes
- Added import method for custom node category icons from ".../sorcar/icons" subdir
- Added Enum property to set default icon style in Preferences
- Added support for Blender icons on nodes of categories: Arrays, Constants, Curves, Flow Control, Inputs, Modifiers, Object Operators, Settings, Utilities
- Added custom base class for node items
- Improved logs for addon updater, node & socket base classes, node-tree, helper methods and operators
- Improved all node classes (from all categories) to call parent (super) class' methods first
- Improved categories layout and nodes sub-layout order of display in "Add Node" menu
- Fixed operators return values
- Renamed icons to match node category names
- Depricated "print_log" helper method

#### v3.2.1

- Added support for object registration & unregistration on each evaluation (handled by nodetree instead of individual nodes)
- Added custom variables in "Custom Python Script" node: _C (bpy context), _D (bpy data), _O (bpy ops), _S (context scene), _N (self node), _NT (self nodetree), _VAR (nodetree variables), _IN (input socket data)
- Added "Select by Index Array" selection node
- Added "Default" input socket in "Get Variable" node to set value if variable not initialised
- Added "Hide Original" bool input prop in "Scatter" node
- Added "Clear Variables" nodetree property in "Properties" UI panel to clear data on re-evaluation
- Added "Clear Preview" operator (shortcut: Alt+E) to clear current preview node
- Added "Execute Node" & "Clear Preview" ops to "Utilities" UI panel
- Improved "Send to Sverchok" node to include selection mask with mesh data sent
- Improved "Receive from Sverchok" node to set selection mask to mesh data received
- Improved "Get/Set Variable" nodes to initialise empty variable in nodetree before accessing property
- Improved "Execute Node" operator: removed redundant tree type check
- Improved "apply_all_modifiers" & "remove_object" helper methods
- Fixed "Separate" node: object array output instead of single object
- Fixed socket interface classes for node-groups
- Fixed "Select Vertices by Connection": switch to vertex mode before selection
- Renamed "Merge" & "Delete" nodes: added "Component" suffix
- Renamed "Cursor" node to "Cursor Transform"
- Renamed "File" input socket to "Directory" in "Export FBX" node

#### v3.2.0

- Added "Node Group" utility node
- Added "Group Nodes" and "Edit Group" operators to create/modify node groups
- Added new keymaps for creating and editing node groups (Ctrl+G, Tab)
- Added UI panel base class and 3 new panels: "Properties", "Utilities" and "Node Groups"
- Added "Send to Sverchok" utility node to pass mesh data (verts/edges/faces) to Sverchok node (Requires "Receive From Sorcar" node in Sverchok nodetree; awaiting PR [#3281](https://github.com/nortikin/sverchok/pull/3281))
- Added "Receive from Sverchok" input node to fetch mesh data (verts/edges/faces) from Sverchok node (Requires "Send To Sorcar" node in Sverchok nodetree; W.I.P.)
- Added "Text Block" constant node to output multi-line string from Text Editor
- Added node interface base class and sub-classes for all sockets to support node-groups (with custom properties)
- Added Wiki and Bug Reporter links in add-ons manager ([#127](https://github.com/aachman98/Sorcar/pull/127))
- Improved object input nodes to clean up geometry when removed
- Improved socket base class to show error icon when execution fails
- Improved "Custom Object" input node: inherit from input base class, apply all modifiers before execution
- Improved Import/Export nodes to select directory path using file picker UI
- Improved "Material Parameter" node to show all available nodes & sockets as a drop-down (search)
- Improved "Scene Info" node: added new outputs for world unit scale length, active/selected/all objects from the scene; removed "Realtime" tree property
- Improved "Duplicate Object" node: added "Linked" input socket
- Improved "get_override" and "sc_poll_op" helper methods
- Improved all operators: show tooltips, new category (prefix)
- Improved all socket base classes to support node-groups
- Fixed socket base class to avoid executing node-group input nodes
- Fixed "Dissolve" node
- Fixed nodetree execution
- Fixed "Array" socket label (length of the evaluated array)
- Fixed "Instancing" node

#### v3.1.6

- Added "Set Dimensions" & "Set Object Name" object operator nodes
- Added "Add ..." input nodes to append primitive object geometry to current object
- Added "set_preview" method in nodetree class
- Added curve ops base class & "Curve Shape/Geometry/Spline Properties" nodes
- Improved "Object Info" node to output object's name, dimensions and bounding box vertices
- Improved "Component Info" node to output face area
- Renamed input nodes to "Create ..." (added prefix)
- Renamed "update_ext" nodetree method to "set_value"
- Fixed "remove_object" helper function
- Fixed vertex group selection & component operator nodes
- Fixed "get_override" helper method to search for correct window/area/region

#### v3.1.5

- Added keymap support to quickly execute selected node (Key: 'E')
- Added new "Curve" category & socket to handle objects with curve data
- Added method for modifying node parameter externally (automatically re-evaluates nodetree)
- Added "Import SVG" & "Text" curve input nodes
- Added "Convert to Mesh" & "Convert to Curve" nodes
- Added "Create Object" input node
- Added "QuadriFlow Remesh" object operator node
- Added "Warp" & "Randomize Vertices" transform nodes
- Added Weld, Lattice, Shrinkwrap & Weighted Normal modifier nodes
- Added "Proportional Editing" & "Snap" settings node
- Added "Clear Parent", "Get Parent", & "Get Children" object operator nodes
- Improved "Scatter" node to support instanced scattering
- Improved "Maths Operation" node to include more operations & better menu layout
- Improved "Parent" node to include option to set the inverse parent correction
- Improved "Skin Modifier" node to allow skin resize for selected vertices
- Renamed transform nodes to "World/Local Transform"
- Fixed issue with Crease, Edge-Slide, Skin-Resize local transform
- Fixed custom object/curve nodes to hide original object
- Fixed issue with rerouting socket connections
- Fixed transform nodes to use snapping & proportional editing settings
- Fixed object deletion helper method to remove orphaned data
- Fixed "Select Nth" node parameter minimum value

#### v3.1.4

- Added "Voxel Remesh" object operator node
- Added "Raycast (Scene)" utility node (renamed "Raycast" to "Raycast (Object)")
- Added "Instancing" & "Parent" object operator nodes
- Added "Empty" and "Single Vertex" input nodes
- Added "Hide/Unhide Component" and "Mark Component" nodes
- Fixed "Element" output pin type of "Begin For-Each Loop" node
- Fixed issue with addon activation
- Fixed int conversion in "Shortest Path" selection nodes

#### v3.1.3

- Added "For-Each Component Loop" nodes
- Added "Get/Set Variable" nodes
- Added "To Sphere" transform node
- Added "Select Vertices by Connections" selection node
- Fixed issue with reset() attribute
- Improved "For" & "For-Each" loop nodes
- Improved "Select Manually" node

#### v3.1.2

- Added noise nodes (cell, fractal, voronoi, ...) in a new category "Noise"
- Added array operation nodes (add, remove, search, ...) in a new category "Arrays"
- Added "Material Parameter" node
- Renamed "Edit Mode" node to "Set Selection Mode"
- Moved "Get Element" & "Make Array" node to "Arrays" category
- Added selection type input socket in "Set Selection Mode" node
- Added custom overridable method for socket layout drawing
- Removed redundant sorcar_updater folder (also added in gitignore)

#### v3.1.1

- Added addon updator by CGCookie
- Fixed issue with realtime update checkbox in "Scene Info" node
- New socket type: Selection Type
- Ability to change selection type directly though selection nodes
- Added issue templates for bug report & feature request

#### v3.1.0

- Support for Blender 2.8
- New architecture for data flow node executions
- Improved socket types and node hierarchy
- Internal data conversions

## Contributors

- [@tomoaki-e33](https://github.com/tomoaki-e33) (Tomoaki Nakano) - Active Developer ([#107](https://github.com/aachman98/Sorcar/pull/107), [#110](https://github.com/aachman98/Sorcar/pull/110), [#111](https://github.com/aachman98/Sorcar/pull/111))
- [@devilvalley](https://github.com/devilvalley) (袁腾鹏) - Bug fix in "Select Nth" selection node ([#106](https://github.com/aachman98/Sorcar/pull/106))
- [@CORPSE-SYS](https://github.com/CORPSE-SYS) - Active Developer ([#83](https://github.com/aachman98/Sorcar/pull/83), [#96](https://github.com/aachman98/Sorcar/pull/96))
- [@zebus3d](https://github.com/zebus3d) - Active Developer ([#88](https://github.com/aachman98/Sorcar/pull/88), [#90](https://github.com/aachman98/Sorcar/pull/90), [#104](https://github.com/aachman98/Sorcar/pull/104))
- [@CGCookie](https://github.com/CGCookie) (CG Cookie) - Addon updater ([Github](https://github.com/CGCookie/blender-addon-updater))
- [@8176135](https://github.com/8176135) - Individual edit mode type in selection nodes using a new socket ([#80](https://github.com/aachman98/Sorcar/pull/80))
- [@BrentARitchie](https://github.com/BrentARitchie) - Documentation Maintainer
- [@huiyao8761380](https://github.com/huiyao8761380) (TangHui) - Documentation Maintainer
- [@Megalomaniak](https://github.com/Megalomaniak) (Felix Kütt) - Documentation Structuring
- [@kichristensen](https://github.com/kichristensen) (Kim Christensen) - Port Sorcar (v2) to Blender 2.80 ([#54](https://github.com/aachman98/Sorcar/pull/54))
- [@SevenNeumann](https://github.com/SevenNeumann) (Mark) - Icons for Sorcar & layout design for main menu ([#46](https://github.com/aachman98/Sorcar/pull/46))

And the amazing [BlenderArtists](https://blenderartists.org) community!
