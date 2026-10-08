import { useState, type FormEvent } from "react";
import {
  DEFAULT_TASK,
  type ExportFormat,
  type HardwareTarget,
  type LayerConfig,
  type TaskConfig,
} from "./types/type";
import { DataSection } from "./components/DataSection";
import { NetworkSection } from "./components/NetworkSection";
import {
  TrainingSection,
  type TrainingParams,
} from "./components/TrainingSection";
import { HardwareSection } from "./components/HardwareSection";
import { ExportSection } from "./components/ExportSection";
import { ProgressPanel } from "./components/ProgressPanel";
import { ModelTester } from "./components/ModelTester";
import { useTrainingSession } from "./hooks/useTrainingSession";
import { summarizeCsv } from "./utils/csv";

export default function App() {
  const [csv, setCsv] = useState("");
  const [layers, setLayers] = useState<LayerConfig>({
    in_features: 3,
    hidden_layers: [{ units: 16, activation: "relu" }],
    out_features: 1,
    output_activation: "none",
  });
  const [task, setTask] = useState<TaskConfig>(DEFAULT_TASK);
  const [params, setParams] = useState<TrainingParams>({
    epochs: 100,
    batch_size: 32,
    lr: 0.001,
  });
  const [hardware, setHardware] = useState<HardwareTarget>({
    target_mode: "CPU",
    device_index: 0,
  });
  const [exportFormat, setExportFormat] = useState<ExportFormat>("json");
  const [outputPath, setOutputPath] = useState("");
  const [formError, setFormError] = useState("");

  const session = useTrainingSession();
  const isClassification = task.loss === "cross_entropy";

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (session.running) return;

    const summary = summarizeCsv(csv);
    if (summary.rows === 0) {
      setFormError("Please load or paste CSV data first.");
      return;
    }
    const expectedColumns = layers.in_features + layers.out_features;
    if (summary.columns !== expectedColumns) {
      setFormError(
        `CSV has ${summary.columns} columns but the network expects ` +
          `${layers.in_features} features + ${layers.out_features} target(s) (${expectedColumns}).`,
      );
      return;
    }

    if (isClassification) {
      if (!Number.isInteger(task.classes) || task.classes < 2) {
        setFormError("Cross-entropy needs at least 2 classes per output (9 for sudoku).");
        return;
      }
      if (!(task.value_scale > 0)) {
        setFormError("Value scale must be greater than 0 (9 for the sudoku CSV).");
        return;
      }
      if (layers.output_activation !== "none") {
        setFormError("Cross-entropy needs the output activation set to none.");
        return;
      }
    }

    setFormError("");
    void session.start({
      config: { ...params, layers, hardware, task },
      data: csv.trim(),
      exportFormat,
      outputPath: outputPath.trim(),
    });
  };

  return (
    <main className="w-full px-[clamp(1.25rem,6vw,15rem)] py-[clamp(1.5rem,3.5vw,4rem)]">
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12 lg:gap-5">
        <header className="bento-tile bento-d1 overflow-hidden rounded-[1.25rem] bg-primary px-6 py-8 text-white sm:px-10 sm:py-10 lg:col-span-12">
          <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
            TensorLoom
          </h1>
          <p className="mt-3 max-w-xl text-sm text-white/70 sm:text-base">
            Train and export neural networks locally.
          </p>
        </header>

        <form onSubmit={handleSubmit} className="contents">
          <div className="bento-tile bento-d2 lg:col-span-5">
            <DataSection
              csv={csv}
              onChange={setCsv}
              onFeatureCountSuggested={(n) =>
                setLayers((prev) => ({ ...prev, in_features: n }))
              }
            />
          </div>

          <div className="bento-tile bento-d3 lg:col-span-7">
            <NetworkSection
              value={layers}
              onChange={setLayers}
              task={task}
              onTaskChange={setTask}
            />
          </div>

          <div className="bento-tile bento-d4 lg:col-span-6">
            <TrainingSection value={params} onChange={setParams} />
          </div>

          <div className="bento-tile bento-d5 lg:col-span-6">
            <HardwareSection value={hardware} onChange={setHardware} />
          </div>

          <div className="bento-tile bento-d6 lg:col-span-9">
            <ExportSection
              format={exportFormat}
              outputPath={outputPath}
              onFormatChange={setExportFormat}
              onOutputPathChange={setOutputPath}
            />
          </div>

          <div className="bento-tile bento-d7 neo-raised flex h-full flex-col justify-between gap-6 p-6 sm:p-8 lg:col-span-3">
            <header>
              <p className="text-xs font-semibold uppercase tracking-[0.14em] text-accent">
                Step 6
              </p>
              <h2 className="mt-2 text-xl font-bold tracking-tight text-primary">
                {session.running ? "Training in progress" : "Ready to train"}
              </h2>
              <p className="mt-2 text-sm leading-relaxed text-secondary">
                {session.running
                  ? "Watch the Progress panel for live loss, network weights, and logs."
                  : "Confirm data, network, and export path, then run a local training session."}
              </p>
            </header>

            <div className="space-y-3">
              <div className="neo-inset flex flex-col gap-2 px-4 py-3 text-sm text-secondary">
                <span className="flex items-center justify-between gap-2">
                  <span>Epochs</span>
                  <span className="font-semibold text-primary">{params.epochs}</span>
                </span>
                <span className="flex items-center justify-between gap-2">
                  <span>Batch</span>
                  <span className="font-semibold text-primary">{params.batch_size}</span>
                </span>
                <span className="flex items-center justify-between gap-2">
                  <span>Loss</span>
                  <span className="font-semibold text-primary">
                    {isClassification ? "cross-entropy" : "mse"}
                  </span>
                </span>
                <span className="flex items-center justify-between gap-2">
                  <span>Device</span>
                  <span className="font-semibold text-primary">
                    {hardware.target_mode}
                  </span>
                </span>
              </div>

              {formError && (
                <p role="alert" className="neo-alert">
                  {formError}
                </p>
              )}

              <button
                type="submit"
                className="neo-btn min-h-14 w-full px-6 py-4 text-base sm:min-h-16 sm:text-lg"
                disabled={session.running}
              >
                {session.running ? "Training..." : "Start training"}
              </button>
            </div>
          </div>
        </form>

        <div className="bento-tile bento-d8 lg:col-span-8">
          <ProgressPanel
            progress={session.progress}
            status={session.status}
            epochs={session.epochs}
            log={session.log}
            snapshot={session.snapshot}
            scoreLabel={isClassification ? "Cell accuracy" : "Score"}
          />
        </div>

        <div className="bento-tile bento-d9 lg:col-span-4">
          <ModelTester suggestedPath={outputPath} />
        </div>
      </div>
    </main>
  );
}