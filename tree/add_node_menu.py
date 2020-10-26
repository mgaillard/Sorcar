menu = [
    ("inputs", [
        "ScEmpty",
        "ScSingleVertex",
        None,
        "ScCreatePlane",
        "ScCreateCube",
        "ScCreateCircle",
        "ScCreateUvSphere",
        "ScCreateIcoSphere",
        "ScCreateCylinder",
        "ScCreateCone",
        "ScCreateTorus",
        "ScCreateMonkey",
        "ScCreateGrid",
        None,
        "ScAddPlane",
        "ScAddCube",
        "ScAddCircle",
        "ScAddUvSphere",
        "ScAddIcoSphere",
        "ScAddCylinder",
        "ScAddCone",
        "ScAddTorus",
        "ScAddMonkey",
        "ScAddGrid",
        None,
        "ScCustomObject",
        "ScCreateObject",
        "ScImportFbx",
        "ScReceiveFromSverchok",
        None,
        "ScCreateAutodiffCube"
    ]),
    ("curves", [
        "ScText",
        None,
        "ScCurveGeometry",
        "ScCurveShape",
        "ScCurveSpline",
        None,
        "ScConvertToCurve",
        "ScConvertToMesh",
        None,
        "ScCustomCurve",
        "ScImportSvg",
    ]),

    None,

    ("transform", [
        "ScWorldTransform",
        "ScLocalTransform",
        None,
        "ScApplyTransform",
        "ScCopyTransform",
        None,
        "ScRandomizeTransform",
        "ScRandomizeVertices",
        None,
        "ScToSphere",
        "ScWarp",
        None,
        "ScCreateOrientation",
        None,
        "ScAutodiffWorldTransform"
    ]),
    ("selection", [
        "ScSelectManually",
        None,
        "ScSelectAll",
        "ScSelectRandom",
        None,
        "ScSelectByIndex",
        "ScSelectByIndexArray",
        "ScSelectByLocation",
        "ScSelectByMaterial",
        "ScSelectByNormal",
        "ScSelectByVertexGroup",
        "ScSelectFaceBySides",
        "ScSelectVerticesByConnections",
        None,
        "ScSelectAlternateFaces",
        "ScSelectNth",
        None,
        "ScSelectLinked",
        "ScSelectLinkedFacesFlat",
        "ScSelectLinkedPick",
        None,
        "ScSelectLoop",
        "ScSelectMultiLoop",
        "ScSelectLoopRegion",
        "ScSelectRegionBoundary",
        None,
        "ScSelectAxis",
        "ScSelectMirror",
        "ScSelectNonManifold",
        "ScSelectSharpEdges",
        None,
        "ScSelectLoose",
        "ScSelectInteriorFaces",
        "ScSelectSimilar",
        "ScSelectSimilarRegion",
        "ScSelectUngrouped",
        None,
        "ScSelectShortestPath",
        "ScSelectShortestPathPick",
        None,
        "ScSelectLess",
        "ScSelectMore",
        None,
        "ScSelectNextItem",
        "ScSelectPrevItem",
    ]),
    ("deletion", [
        "ScDeleteComponents",
        "ScDeleteEdgeLoop",
        "ScDeleteLoose",
        None,
        "ScDissolve",
        "ScDissolveDegenerate",
        "ScDissolveEdges",
        "ScDissolveFaces",
        "ScDissolveLimited",
        "ScDissolveVertices",
        None,
        "ScEdgeCollapse",
    ]),

    None,

    ("component_operators", [
        "ScBevel",
        "ScInset",
        "ScPoke",
        None,
        "ScExtrude",
        "ScExtrudeEdges",
        "ScExtrudeFaces",
        "ScExtrudeVertices",
        "ScExtrudeRegion",
        None,
        "ScFlatten",
        "ScRotateEdge",
        "ScScrew",
        "ScSpin",
        "ScSolidify",
        "ScWireframe",
        None,
        "ScMergeComponents",
        "ScMergeNormals",
        None,
        "ScSplit",
        "ScSeparate",
        "ScRip",
        "ScRipEdge",
        None,
        "ScBisect",
        "ScConvexHull",
        "ScLoopCut",
        "ScOffsetEdgeLoops",
        None,
        "ScSymmetrize",
        "ScSymmetrySnap",
        None,
        "ScAddEdgeFace",
        "ScConnectVertices",
        "ScBridgeEdgeLoops",
        "ScBeautifyFill",
        "ScFillEdgeLoop",
        "ScFillGrid",
        "ScFillHoles",
        None,
        "ScAverageNormals",
        "ScMakeNormalsConsistent",
        "ScFlipNormals",
        "ScPointNormals",
        None,
        "ScDecimate",
        "ScUnsubdivide",
        "ScRemoveDoubles",
        None,
        "ScSubdivide",
        "ScSubdivideEdgeRing",
        None,
        "ScMarkComponent",
        "ScDuplicateComponent",
        "ScHideComponents",
        "ScUnhideComponents",
        None,
        "ScIntersect",
        "ScIntersectBoolean",
        None,
        "ScQuadrangulate",
        "ScTriangulate",
        None,
        "ScMaterial",
        "ScVertexGroup",
        None,
        "ScUvProject",
        "ScUvSmartProject",
        "ScUvUnwrap",
    ]),
    ("object_operators", [
        "ScSetName",
        "ScSetDimensions",
        "ScOrigin",
        None,
        "ScShading",
        "ScDrawMode",
        None,
        "ScMergeObjects",
        "ScDuplicateObject",
        None,
        "ScParent",
        "ScClearParent",
        "ScGetParent",
        "ScGetChildren",
        None,
        "ScFindNearest",
        "ScOverlap",
        "ScRaycastObject",
        None,
        "ScInstancing",
        "ScMakeLinks",
        "ScScatter",
        None,
        "ScVoxelRemesh",
        "ScQuadriFlowRemesh",
        None,
        "ScExportFbx",
    ]),
    ("modifiers", [
        "ScArrayMod",
        "ScBevelMod",
        "ScBooleanMod",
        "ScBuildMod",
        "ScDecimateMod",
        "ScEdgeSplitMod",
        "ScMirrorMod",
        "ScRemeshMod",
        "ScScrewMod",
        "ScSkinMod",
        "ScSolidifyMod",
        "ScSubsurfMod",
        "ScTriangulateMod",
        "ScWeldMod",
        "ScWireframeMod",
        None,
        "ScCastMod",
        "ScCorrectiveSmoothnessMod",
        "ScCurveMod",
        "ScDisplaceMod",
        "ScHookMod",
        "ScLaplacianSmoothMod",
        "ScLatticeMod",
        "ScShrinkwrapMod",
        "ScSimpleDeformMod",
        "ScSmoothMod",
        "ScWaveMod",
        None,
        "ScWeightedNormalMod",
    ]),

    None,

    ("constants", [
        "ScNumber",
        "ScVector",
        "ScBool",
        None,
        "ScAutodiffNumber",
        None,
        "ScString",
        "ScTextBlock",
        None,
        "ScSelectionType",
    ]),
    ("arrays", [
        "ScMakeArray",
        None,
        "ScAddElement",
        "ScAddArray",
        None,
        "ScRemoveElement",
        "ScPopElement",
        "ScReverseArray",
        "ScClearArray",
        None,
        "ScGetElement",
        "ScCountElement",
        "ScSearchElement",

    ]),
    ("noise", [
        "ScCell",
        "ScFractal",
        "ScHeteroTerrain",
        "ScHybridMultiFractal",
        "ScMultiFractal",
        "ScNoise",
        "ScRidgedMultiFractal",
        "ScTurbulence",
        "ScVariableLacunarity",
        "ScVoronoi",
        None,
        "ScCellVector",
        "ScNoiseVector",
        "ScTurbulenceVector",
    ]),

    None,

    ("utilities", [
        "ScAppendString",
        None,
        "ScMathsOp",
        "ScTrigoOp",
        "ScClamp",
        "ScMapRange",
        None,
        "ScVectorOp",
        "ScBreakVector",
        None,
        "ScBooleanOp",
        "ScComparisonOp",
        None,
        "ScComponentInfo",
        "ScObjectInfo",
        "ScSceneInfo",
        None,
        "ScCustomPythonScript",
        "ScEvaluateAs",
        "ScPrint",
        None,
        "ScGetVariable",
        "ScSetVariable",
        None,
        "ScMaterialParameter",
        "ScRaycastScene",
        "ScSendToSverchok",
        None,
        "ScNodeGroup",
        None,
        "ScAutodiffNumberConvert",
        "ScAutodiffNumberDerivative",
    ]),
    ("settings", [
        "ScSetSelectionMode",
        "ScPivotPoint",
        "ScCursorTransform",
        "ScSnap",
        "ScProportionalEditing",
        "ScTransformOrientation",
    ]),

    None,

    ("flow_control", [
        "ScBranch",
        None,
        "ScBeginForLoop",
        "ScEndForLoop",
        None,
        "ScBeginForEachLoop",
        "ScEndForEachLoop",
        None,
        "ScBeginForEachComponentLoop",
        "ScEndForEachComponentLoop",
    ])
]