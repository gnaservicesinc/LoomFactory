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
                    .keyboardShortcut("o").disabled(model.busy)
                Button("Import Images…") { model.importImages() }
                    .keyboardShortcut("i").disabled(model.snapshot == nil || model.busy)
            }
            CommandGroup(after: .pasteboard) {
                Button("Manage Models") { model.workspaceSection = .models }
                    .keyboardShortcut("m", modifiers: [.command, .shift])
                    .disabled(model.snapshot == nil)
                Button("Photo Training") { model.workspaceSection = .photoTraining }
                    .keyboardShortcut("p", modifiers: [.command, .shift])
                    .disabled(model.snapshot == nil)
                Button("Refresh Source Folder") { model.refresh() }
                    .keyboardShortcut("r").disabled(model.snapshot == nil || model.busy)
            }
            CommandGroup(replacing: .saveItem) {
                Button("Save Checkpoint") { model.saveCheckpoint() }
                    .keyboardShortcut("s")
                    .disabled(!model.isTraining)
                Button("Save & Stop Training") { model.saveAndStopTraining() }
                    .disabled(!model.isTraining)
            }
        }
    }
}

@MainActor
final class FactoryApplicationDelegate: NSObject, NSApplicationDelegate {
    var operationIsActive: () -> Bool = { false }
    var trainingIsActive: () -> Bool = { false }
    var saveAndStopForTermination: (() async -> Bool)?
    var chooseTrainingQuit: () -> Bool = {
        let alert = NSAlert()
        alert.messageText = "Save training before quitting?"
        alert.informativeText = "Save & Quit saves the latest completed update and its optimizer state, then stops training. If saving fails, Loom Factory stays open."
        alert.addButton(withTitle: "Keep Training")
        alert.addButton(withTitle: "Save & Quit")
        return alert.runModal() == .alertSecondButtonReturn
    }
    var showBusyNotice: () -> Void = {
        let alert = NSAlert()
        alert.messageText = "An operation is still running"
        alert.informativeText = "Finish or stop the current operation in Loom Room before quitting."
        alert.addButton(withTitle: "Keep Working")
        alert.runModal()
    }
    var prepareForTermination: (() async -> Bool)?
    var replyToTermination: (NSApplication, Bool) -> Void = { application, allowed in
        application.reply(toApplicationShouldTerminate: allowed)
    }
    private var terminationPending = false

    func bind(to model: FactoryModel) {
        operationIsActive = { [weak model] in model?.busy ?? false }
        trainingIsActive = { [weak model] in model?.isTraining ?? false }
        saveAndStopForTermination = { [weak model] in
            guard let model else { return true }
            return await model.saveTrainingAndPrepareForTermination()
        }
        prepareForTermination = { [weak model] in
            guard let model else { return true }
            return await model.prepareForTermination()
        }
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard !terminationPending else { return .terminateLater }
        if operationIsActive() {
            guard trainingIsActive(), let saveAndStopForTermination else {
                showBusyNotice()
                return .terminateCancel
            }
            guard chooseTrainingQuit() else { return .terminateCancel }
            terminationPending = true
            Task { @MainActor [weak self] in
                let saved = await saveAndStopForTermination()
                guard let self else { return }
                self.terminationPending = false
                self.replyToTermination(sender, saved && !self.operationIsActive())
            }
            return .terminateLater
        }
        guard let prepareForTermination else { return .terminateNow }
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
