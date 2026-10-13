// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "LoomFactory",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "LoomCore", targets: ["LoomCore"]),
        .executable(name: "LoomFactory", targets: ["LoomFactory"])
    ],
    targets: [
        .target(name: "LoomCore"),
        .executableTarget(name: "LoomFactory", dependencies: ["LoomCore"]),
        .testTarget(name: "LoomCoreTests", dependencies: ["LoomCore"], path: "Tests/LoomCoreTests"),
        .testTarget(name: "LoomFactoryUITests", dependencies: ["LoomFactory", "LoomCore"], path: "Tests/LoomFactoryUITests")
    ]
)
