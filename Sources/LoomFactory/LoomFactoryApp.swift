import AppKit
import SwiftUI

@main
struct LoomFactoryApp: App {
    @NSApplicationDelegateAdaptor(FactoryApplicationDelegate.self) private var applicationDelegate
    @State private var model = FactoryModel()

    var body: some Scene {
        WindowGroup("Loom Factory", id: "loom-room") {
            LoomRoomView(model: model)
                .frame(minWidth: 1120, minHeight: 760)
                .tint(FactoryStyle.accent)
                .onAppear { applicationDelegate.bind(to: model) }
        }
        .defaultSize(width: 1380, height: 980)
        .windowToolbarStyle(.unified)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Open Source Folder…") { model.chooseSourceFolder() }
                    .keyboardShortcut("o")
                Button("Import Images…") { model.importImages() }
                    .keyboardShortcut("i").disabled(model.snapshot == nil || model.busy)
            }
            CommandGroup(after: .pasteboard) {
                Button("Refresh Source Folder") { model.refresh() }
                    .keyboardShortcut("r").disabled(model.snapshot == nil || model.busy)
            }
        }
    }
}

@MainActor
final class FactoryApplicationDelegate: NSObject, NSApplicationDelegate {
    var operationIsActive: () -> Bool = { false }
    var prepareForTermination: (() async -> Bool)?
    var replyToTermination: (NSApplication, Bool) -> Void = { application, allowed in
        application.reply(toApplicationShouldTerminate: allowed)
    }
    private var terminationPending = false

    func bind(to model: FactoryModel) {
        operationIsActive = { [weak model] in model?.busy ?? false }
        prepareForTermination = { [weak model] in
            guard let model else { return true }
            return await model.prepareForTermination()
        }
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard !operationIsActive() else {
            NSLog("Loom Factory is busy. Stop the current operation in Loom Room before quitting.")
            return .terminateCancel
        }
        guard let prepareForTermination else { return .terminateNow }
        guard !terminationPending else { return .terminateLater }
        terminationPending = true
        Task { @MainActor [weak self] in
            let prepared = await prepareForTermination()
            guard let self else { return }
            self.terminationPending = false
            self.replyToTermination(sender, prepared && !self.operationIsActive())
        }
        return .terminateLater
    }
}
