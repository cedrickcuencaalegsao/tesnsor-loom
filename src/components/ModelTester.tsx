import { useRef, useState, type ChangeEvent } from "react";
import {
  evaluateModel,
  loadModel,
  pickModelFile,
  predictRows,
  type Evaluation,
  type ModelSummary,
} from "../api/model";

// Classification models (softmax + cross-entropy) also report accuracy figures.
// Declared here so this file works even if api/model.ts has not been updated yet.
type ScoredEvaluation = Evaluation & {
  cell_accuracy?: number | null;
  board_accuracy?: number | null;
  blank_accuracy?: number | null;
};

interface Props {
  /** Optional: the export path from the training form, offered as a shortcut. */
  suggestedPath?: string;
}

function describeScore(r2: number): string {
  if (r2 < 0)
    return "Worse than always guessing the average. The model needs more training (try a higher learning rate or more epochs).";
  if (r2 < 0.5)
    return "Weak. It has learned a little, but a lot of the pattern is missing.";
  if (r2 < 0.9) return "Decent. It captures most of the pattern.";
  return "Strong. Predictions track the targets closely.";
}

function describeAccuracy(acc: number, board: number | null | undefined): string {
  const boardNote =
    board != null && board < 0.5
      ? " Whole-board accuracy is low, so use it as a helper (fill the most confident cell, repeat) or pair it with a solver."
      : "";
  if (acc < 0.3)
    return "Weak. Barely better than guessing; it needs more data or training." + boardNote;
  if (acc < 0.7) return "Decent. It has learned part of the pattern." + boardNote;
  if (acc < 0.95) return "Good. Most cells are right." + boardNote;
  return "Strong. Nearly every cell is right." + boardNote;
}

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

/** Format decoded digits; 81 values are laid out as a 9x9 grid. */
function formatDigits(values: number[], scale: number): string {
  const digits = values.map((v) => Math.round(v * scale));
  if (digits.length === 81) {
    const rows: string[] = [];
    for (let r = 0; r < 9; r++) {
      const row = digits.slice(r * 9, r * 9 + 9);
      rows.push(
        [row.slice(0, 3), row.slice(3, 6), row.slice(6, 9)]
          .map((g) => g.join(" "))
          .join(" | "),
      );
      if (r === 2 || r === 5) rows.push("------+-------+------");
    }
    return rows.join("\n");
  }
  return digits.join(", ");
}

export function ModelTester({ suggestedPath }: Props) {
  const [summary, setSummary] = useState<ModelSummary | null>(null);
  const [error, setError] = useState("");

  const [inputText, setInputText] = useState("");
  const [prediction, setPrediction] = useState<number[] | null>(null);
  // Classification models output class / scale; set this to the training "Value scale" (9 for sudoku).
  const [decodeScale, setDecodeScale] = useState("");

  const [evalCsv, setEvalCsv] = useState("");
  const [evalFileName, setEvalFileName] = useState("");
  const [evaluation, setEvaluation] = useState<ScoredEvaluation | null>(null);
  const [busy, setBusy] = useState(false);
  const evalFileRef = useRef<HTMLInputElement>(null);

  const canUseSuggested =
    !!suggestedPath &&
    (suggestedPath.endsWith(".json") || suggestedPath.endsWith(".bin"));

  const openModel = async (path: string) => {
    setError("");
    setPrediction(null);
    setEvaluation(null);
    try {
      setSummary(await loadModel(path));
    } catch (err) {
      setSummary(null);
      setError(String(err));
    }
  };

  const handleBrowse = async () => {
    try {
      const path = await pickModelFile();
      if (path) await openModel(path);
    } catch (err) {
      setError(`Could not open the file dialog: ${String(err)}`);
    }
  };

  const handlePredict = async () => {
    if (!summary) return;
    setError("");
    setPrediction(null);

    const values = inputText
      .split(/[,\s]+/)
      .filter((s) => s.length > 0)
      .map(Number);

    if (values.length !== summary.input_size || values.some((v) => !Number.isFinite(v))) {
      setError(`Enter exactly ${summary.input_size} numbers, separated by commas.`);
      return;
    }

    setBusy(true);
    try {
      const [out] = await predictRows(summary.path, [values]);
      setPrediction(out);
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  };

  const handleEvalFile = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setEvalCsv(await file.text());
    setEvalFileName(file.name);
    setEvaluation(null);
  };

  const handleEvaluate = async () => {
    if (!summary || !evalCsv) return;
    setError("");
    setBusy(true);
    try {
      setEvaluation(await evaluateModel(summary.path, evalCsv));
    } catch (err) {
      setEvaluation(null);
      setError(String(err));
    } finally {
      setBusy(false);
    }
  };

  const placeholder = summary
    ? Array.from({ length: summary.input_size }, (_, i) =>
        (0.1 * (i + 1)).toFixed(1),
      ).join(", ")
    : "";

  const scale = Number(decodeScale);
  const decode = decodeScale.trim() !== "" && Number.isFinite(scale) && scale > 0;
  const isClassification = evaluation?.cell_accuracy != null;

  return (
    <section className="neo-raised h-full w-full p-6 sm:p-8">
      <header className="mb-6">
        <h2 className="text-xl font-bold tracking-tight text-primary">
          Test a trained model
        </h2>
        <p className="mt-1 text-sm text-secondary">
          Load a model to predict a row or score a CSV.
        </p>
      </header>

      <div className="neo-inset flex flex-wrap items-center justify-between gap-3 px-4 py-4">
        <span className="text-sm font-medium text-secondary">Model file</span>
        <div className="flex flex-wrap gap-2">
          <button type="button" className="neo-btn" onClick={handleBrowse}>
            Open model...
          </button>
          {canUseSuggested && (
            <button
              type="button"
              className="neo-btn-ghost"
              onClick={() => openModel(suggestedPath!)}
            >
              Use export path
            </button>
          )}
        </div>
      </div>

      {error && (
        <p role="alert" className="neo-alert mt-4">
          {error}
        </p>
      )}

      {summary && (
        <div className="mt-6 space-y-8">
          <div className="space-y-3">
            <h3 className="text-sm font-semibold text-primary">Model</h3>
            <p className="break-all text-xs text-secondary">{summary.path}</p>
            <div className="flex flex-wrap gap-2">
              <span className="neo-chip">{summary.input_size} inputs</span>
              <span className="neo-chip">{summary.output_size} output(s)</span>
              <span className="neo-chip">{summary.parameters} parameters</span>
            </div>
            <ol className="neo-inset space-y-2 px-4 py-3 text-sm text-secondary">
              {summary.layers.map((layer, i) => (
                <li key={i}>
                  {layer.inputs} to {layer.outputs} ({layer.activation})
                </li>
              ))}
            </ol>
          </div>

          <div className="space-y-3">
            <h3 className="text-sm font-semibold text-primary">Predict one row</h3>
            <label
              htmlFor="predict-features"
              className="block text-sm font-medium text-secondary"
            >
              Features ({summary.input_size} values, comma separated)
            </label>
            <div className="neo-inset flex flex-col gap-3 px-4 py-4 sm:flex-row sm:items-center">
              <input
                id="predict-features"
                type="text"
                value={inputText}
                placeholder={placeholder}
                onChange={(e) => setInputText(e.target.value)}
                className="neo-field min-w-0 flex-1 border-0 shadow-none"
              />
              <button
                type="button"
                className="neo-btn shrink-0"
                onClick={handlePredict}
                disabled={busy}
              >
                Predict
              </button>
            </div>

            <div className="space-y-1.5">
              <label
                htmlFor="decode-scale"
                className="block text-xs font-medium text-secondary"
              >
                Show result as whole numbers: multiply by (classification models, e.g. 9)
              </label>
              <input
                id="decode-scale"
                type="number"
                min={0}
                step="any"
                value={decodeScale}
                placeholder="off"
                onChange={(e) => setDecodeScale(e.target.value)}
                className="neo-field"
              />
            </div>

            {prediction && !decode && (
              <p className="neo-chip">
                Prediction: {prediction.map((v) => v.toFixed(4)).join(", ")}
              </p>
            )}
            {prediction && decode && (
              <pre className="neo-inset neo-log">{formatDigits(prediction, scale)}</pre>
            )}
          </div>

          <div className="space-y-3">
            <h3 className="text-sm font-semibold text-primary">Score on a CSV</h3>
            <div className="neo-inset flex flex-wrap items-center justify-between gap-3 px-4 py-4">
              <span className="text-sm font-medium text-secondary">
                CSV with a target column (last column)
              </span>
              <input
                ref={evalFileRef}
                type="file"
                accept=".csv,text/csv,text/plain"
                className="sr-only"
                onChange={handleEvalFile}
              />
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  className="neo-btn-ghost"
                  onClick={() => evalFileRef.current?.click()}
                >
                  Choose CSV
                </button>
                <button
                  type="button"
                  className="neo-btn"
                  onClick={handleEvaluate}
                  disabled={busy || !evalCsv}
                >
                  Evaluate
                </button>
              </div>
            </div>
            {evalFileName && (
              <span className="neo-chip">Loaded {evalFileName}</span>
            )}
            {evaluation && (
              <div className="space-y-3">
                <div className="neo-inset overflow-x-auto px-2 py-2">
                  <table className="neo-table">
                    <tbody>
                      <tr>
                        <th>Rows</th>
                        <td>{evaluation.rows}</td>
                      </tr>
                      {isClassification ? (
                        <>
                          {evaluation.blank_accuracy != null && (
                            <tr>
                              <th>Blank-cell accuracy</th>
                              <td>{pct(evaluation.blank_accuracy)}</td>
                            </tr>
                          )}
                          <tr>
                            <th>Cell accuracy (incl. given clues)</th>
                            <td>{pct(evaluation.cell_accuracy ?? 0)}</td>
                          </tr>
                          <tr>
                            <th>Full-board accuracy</th>
                            <td>{pct(evaluation.board_accuracy ?? 0)}</td>
                          </tr>
                        </>
                      ) : (
                        <>
                          <tr>
                            <th>Mean squared error</th>
                            <td>{evaluation.mse.toFixed(4)}</td>
                          </tr>
                          <tr>
                            <th>R² score</th>
                            <td>{evaluation.r2.toFixed(3)}</td>
                          </tr>
                          <tr>
                            <th>Mean target</th>
                            <td>{evaluation.mean_target.toFixed(4)}</td>
                          </tr>
                        </>
                      )}
                    </tbody>
                  </table>
                </div>
                <p className="text-sm text-secondary">
                  {isClassification
                    ? describeAccuracy(
                        evaluation.blank_accuracy ?? evaluation.cell_accuracy ?? 0,
                        evaluation.board_accuracy,
                      )
                    : describeScore(evaluation.r2)}
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  );
}