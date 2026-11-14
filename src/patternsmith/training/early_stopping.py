import math


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
    ):
        if patience <= 0:
            raise ValueError("patience must be greater than zero.")

        if not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be a non-negative finite number.")

        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, validation_loss: float) -> bool:
        """Update early-stopping state and return whether training should stop."""

        if not math.isfinite(validation_loss):
            self.counter += 1
        elif validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

        return self.should_stop

    def state_dict(self) -> dict:
        """Return the current early-stopping state."""

        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_loss": self.best_loss,
            "counter": self.counter,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore early-stopping state."""

        required_keys = {
            "patience",
            "min_delta",
            "best_loss",
            "counter",
            "should_stop",
        }

        missing_keys = required_keys.difference(state)

        if missing_keys:
            raise ValueError(
                f"Invalid early-stopping state. Missing keys: "
                f"{sorted(missing_keys)}"
            )

        patience = state["patience"]
        min_delta = state["min_delta"]
        best_loss = state["best_loss"]
        counter = state["counter"]
        should_stop = state["should_stop"]

        if not isinstance(patience, int) or patience <= 0:
            raise ValueError("Invalid patience in early-stopping state.")

        if not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("Invalid min_delta in early-stopping state.")

        if counter < 0:
            raise ValueError("Invalid counter in early-stopping state.")

        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = best_loss
        self.counter = counter
        self.should_stop = should_stop
