import type { NetworkSnapshot } from "../types/type";
import type { EpochRow } from "../hooks/useTrainingSession";
import { NetworkVisualizer } from "./NetworkVisualizer";
import { LossChart } from "./LossChart";

interface Props {
  progress: number; // 0..1
  status: string;
  epochs: EpochRow[];
  log: string[];
  snapshot: NetworkSnapshot | null;
  /** Column title for the validation score ("Score" for regression, "Cell accuracy" for classification). */
  scoreLabel?: string;
}

const VISIBLE_ROWS = 15;

export function ProgressPanel({
  progress,
  status,
  epochs,
  log,
  snapshot,
  scoreLabel = "Score",
}: Props) {
  const pct = Math.round(Math.max(0, Math.min(1, progress)) * 100);
  const recent = epochs.slice(-VISIBLE_ROWS);

  return (
    <section className="neo-raised h-full w-full p-6 sm:p-8">
      <header className="mb-6">
        <h2 className="text-xl font-bold tracking-tight text-primary">Progress</h2>
        <p className="mt-1 text-sm text-secondary">
          Live training status, network view, and loss.
        </p>
      </header>

      <div className="space-y-2">
        <div className="flex items-center justify-between gap-3">
          <span className="text-sm font-medium text-secondary">Completion</span>
          <span className="neo-chip">{pct}%</span>
        </div>
        <progress className="neo-progress" value={pct} max={100} />
        {status && (
          <p className="text-sm text-secondary">{status}</p>
        )}
      </div>

      <div className="mt-8 grid grid-cols-1 gap-6 sm:grid-cols-2 sm:gap-4">
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-primary">Network (live)</h3>
          <div className="neo-inset overflow-x-auto p-4">
            <NetworkVisualizer snapshot={snapshot} />
          </div>
        </div>
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-primary">Loss</h3>
          <div className="neo-inset overflow-x-auto p-4">
            <LossChart values={epochs.map((row) => row.avgLoss)} />
          </div>
        </div>
      </div>

      <div className="mt-8 space-y-3">
        <h3 className="text-sm font-semibold text-primary">Recent epochs</h3>
        <div className="neo-inset overflow-x-auto px-2 py-2">
          <table className="neo-table">
            <thead>
              <tr>
                <th>Epoch</th>
                <th>Avg loss</th>
                <th>{scoreLabel}</th>
              </tr>
            </thead>
            <tbody>
              {recent.length === 0 ? (
                <tr>
                  <td colSpan={3} className="py-4 text-center text-secondary">
                    No epochs yet.
                  </td>
                </tr>
              ) : (
                recent.map((row) => (
                  <tr key={row.epoch}>
                    <td>{row.epoch}</td>
                    <td>{row.avgLoss.toFixed(6)}</td>
                    <td>{row.score.toFixed(4)}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
        {epochs.length > VISIBLE_ROWS && (
          <p className="text-xs text-secondary">
            Showing last {VISIBLE_ROWS} of {epochs.length} epochs.
          </p>
        )}
      </div>

      <div className="mt-8 space-y-3">
        <h3 className="text-sm font-semibold text-primary">Log</h3>
        <pre className="neo-inset neo-log">{log.join("\n") || "No log entries yet."}</pre>
      </div>
    </section>
  );
}