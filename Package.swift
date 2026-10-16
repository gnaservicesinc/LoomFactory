// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "LoomFactory",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "LoomNative", targets: ["LoomNative"]),
        .executable(name: "LoomNativeProbe", targets: ["LoomNativeProbe"]),
        .executable(name: "LoomTrainingQueue", targets: ["LoomTrainingQueue"]),
        .library(name: "LoomCore", targets: ["LoomCore"]),
        .executable(name: "LoomFactory", targets: ["LoomFactory"])
    ],
    targets: [
        .target(name: "LoomNative"),
        .executableTarget(name: "LoomNativeProbe", dependencies: ["LoomNative"]),
        .executableTarget(name: "LoomTrainingQueue", dependencies: ["LoomCore", "LoomNative"]),
        .target(name: "LoomCore", dependencies: ["LoomNative"], resources: [.copy("Resources/BlenderExport.py")]),
        .executableTarget(name: "LoomFactory", dependencies: ["LoomCore"]),
        .testTarget(name: "LoomNativeTests", dependencies: ["LoomNative"], path: "Tests/LoomNativeTests"),
        .testTarget(name: "LoomCoreTests", dependencies: ["LoomCore", "LoomNative"], path: "Tests/LoomCoreTests"),
        .testTarget(name: "LoomFactoryUITests", dependencies: ["LoomFactory", "LoomCore"], path: "Tests/LoomFactoryUITests")
    ]
)
