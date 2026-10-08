import type {
  HiddenLayerConfig,
  LayerConfig,
  LossKind,
  TaskConfig,
} from "../types/type";

interface Props {
  value: LayerConfig;
  onChange: (value: LayerConfig) => void;
  task: TaskConfig;
  onTaskChange: (value: TaskConfig) => void;
}

const MAX_HIDDEN_LAYERS = 8;
const HIDDEN_ACTIVATIONS = ["relu", "tanh", "sigmoid"] as const;
const OUTPUT_ACTIVATIONS = ["none", "relu", "tanh", "sigmoid"] as const;
const LOSS_OPTIONS = ["mse", "cross_entropy"] as const;

function ActivationPicker({
  id,
  label,
  options,
  value,
  onChange,
}: {
  id: string;
  label: string;
  options: readonly string[];
  value: string;
  onChange: (next: string) => void;
}) {
  return (
    <div className="space-y-1.5">
      <span id={id} className="block text-xs font-medium text-secondary">
        {label}
      </span>
      <div
        role="group"
        aria-labelledby={id}
        className="neo-segment"
      >
        {options.map((option) => {
          const selected = value === option;
          return (
            <button
              key={option}
              type="button"
              aria-pressed={selected}
              className="neo-segment-option"
              onClick={() => onChange(option)}
            >
              {option}
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function NetworkSection({ value, onChange, task, onTaskChange }: Props) {
  const isClassification = task.loss === "cross_entropy";

  const setHidden = (index: number, patch: Partial<HiddenLayerConfig>) => {
    const hidden_layers = value.hidden_layers.map((layer, i) =>
      i === index ? { ...layer, ...patch } : layer,
    );
    onChange({ ...value, hidden_layers });
  };

  const addHidden = () => {
    if (value.hidden_layers.length >= MAX_HIDDEN_LAYERS) return;
    const last = value.hidden_layers[value.hidden_layers.length - 1];
    onChange({
      ...value,
      hidden_layers: [
        ...value.hidden_layers,
        { units: last?.units ?? 16, activation: last?.activation ?? "relu" },
      ],
    });
  };

  const removeHidden = (index: number) => {
    onChange({
      ...value,
      hidden_layers: value.hidden_layers.filter((_, i) => i !== index),
    });
  };

  const setLoss = (loss: LossKind) => {
    if (loss === task.loss) return;
    if (loss === "cross_entropy") {
      // Raw logits out of the network: softmax happens inside the loss.
      onTaskChange({
        loss,
        classes: task.classes >= 2 ? task.classes : 9,
        value_scale: task.value_scale > 1 ? task.value_scale : 9,
      });
      onChange({ ...value, output_activation: "none" });
    } else {
      onTaskChange({ loss, classes: 0, value_scale: 1 });
    }
  };

  const networkIn = isClassification
    ? value.in_features * (task.classes + 1)
    : value.in_features;
  const networkOut = isClassification
    ? value.out_features * task.classes
    : value.out_features;

  return (
    <section className="neo-raised h-full w-full p-6 sm:p-8">
      <header className="mb-6">
        <h2 className="text-xl font-bold tracking-tight text-primary">2. Network</h2>
        <p className="mt-1 text-sm text-secondary">
          Configure input, hidden, and output layers.
        </p>
      </header>

      <div className="mb-6 space-y-3">
        <ActivationPicker
          id="loss-kind"
          label="Task / loss"
          options={LOSS_OPTIONS}
          value={task.loss}
          onChange={(next) => setLoss(next as LossKind)}
        />
        {isClassification ? (
          <div className="neo-inset space-y-3 px-4 py-4">
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <div className="space-y-1.5">
                <label
                  htmlFor="task-classes"
                  className="block text-xs font-medium text-secondary"
                >
                  Classes per output
                </label>
                <input
                  id="task-classes"
                  type="number"
                  min={2}
                  step={1}
                  required
                  value={task.classes}
                  onChange={(e) =>
                    onTaskChange({ ...task, classes: Number(e.target.value) })
                  }
                  className="neo-field"
                />
              </div>
              <div className="space-y-1.5">
                <label
                  htmlFor="task-scale"
                  className="block text-xs font-medium text-secondary"
                >
                  Value scale
                </label>
                <input
                  id="task-scale"
                  type="number"
                  min={0}
                  step="any"
                  required
                  value={task.value_scale}
                  onChange={(e) =>
                    onTaskChange({ ...task, value_scale: Number(e.target.value) })
                  }
                  className="neo-field"
                />
              </div>
            </div>
            <p className="text-xs text-secondary">
              Each CSV value is class / scale (sudoku: 9 classes, scale 9; inputs use 0 for
              blank, targets are 1 to 9). Inputs are one-hot encoded internally, so the
              network is {networkIn} inputs to {networkOut} outputs.
            </p>
          </div>
        ) : (
          <p className="text-xs text-secondary">
            Mean squared error: predicts numbers (regression).
          </p>
        )}
      </div>

      <div className="space-y-2">
        <label
          htmlFor="in-features"
          className="block text-sm font-medium text-secondary"
        >
          Input features
        </label>
        <input
          id="in-features"
          type="number"
          min={1}
          step={1}
          required
          value={value.in_features}
          onChange={(e) =>
            onChange({ ...value, in_features: Number(e.target.value) })
          }
          className="neo-field"
        />
        <p className="text-xs text-secondary">
          Must equal the number of CSV feature columns.
        </p>
      </div>

      <div className="mt-6">
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <h3 className="text-sm font-semibold text-primary">Hidden layers</h3>
          <span className="neo-chip">{value.hidden_layers.length}</span>
        </div>

        {value.hidden_layers.length === 0 && (
          <p className="mb-3 text-sm text-secondary">
            No hidden layers: the network is a single linear layer from input to
            output.
          </p>
        )}

        <div className="space-y-3">
          {value.hidden_layers.map((layer, i) => (
            <div
              key={i}
              className="neo-inset space-y-3 px-4 py-4"
            >
              <p className="text-sm font-semibold text-primary">
                Hidden {i + 1}
              </p>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <div className="space-y-1.5">
                  <label
                    htmlFor={`hidden-units-${i}`}
                    className="block text-xs font-medium text-secondary"
                  >
                    Units
                  </label>
                  <input
                    id={`hidden-units-${i}`}
                    type="number"
                    min={1}
                    step={1}
                    required
                    value={layer.units}
                    onChange={(e) =>
                      setHidden(i, { units: Number(e.target.value) })
                    }
                    className="neo-field"
                  />
                </div>
                <ActivationPicker
                  id={`hidden-act-${i}`}
                  label="Activation"
                  options={HIDDEN_ACTIVATIONS}
                  value={layer.activation}
                  onChange={(activation) => setHidden(i, { activation })}
                />
              </div>
              <button
                type="button"
                className="neo-btn-ghost"
                onClick={() => removeHidden(i)}
              >
                Remove
              </button>
            </div>
          ))}
        </div>

        <div className="mt-4">
          <button
            type="button"
            className="neo-btn"
            onClick={addHidden}
            disabled={value.hidden_layers.length >= MAX_HIDDEN_LAYERS}
          >
            + Add hidden layer
          </button>
        </div>
      </div>

      <div className="mt-8 space-y-4">
        <h3 className="text-sm font-semibold text-primary">Output layer</h3>

        <div className="space-y-2">
          <label
            htmlFor="out-features"
            className="block text-sm font-medium text-secondary"
          >
            Output units
          </label>
          <input
            id="out-features"
            type="number"
            min={1}
            step={1}
            required
            value={value.out_features}
            onChange={(e) =>
              onChange({ ...value, out_features: Number(e.target.value) })
            }
            className="neo-field"
          />
          <p className="text-xs text-secondary">
            The last this-many CSV columns are the targets.
          </p>
        </div>

        <div className="space-y-2">
          <ActivationPicker
            id="out-activation"
            label="Output activation"
            options={isClassification ? ["none"] : OUTPUT_ACTIVATIONS}
            value={value.output_activation}
            onChange={(output_activation) =>
              onChange({ ...value, output_activation })
            }
          />
          <p className="text-xs text-secondary">
            {isClassification
              ? "Raw scores (logits): softmax is applied inside the loss."
              : "Use none for regression."}
          </p>
        </div>
      </div>
    </section>
  );
}