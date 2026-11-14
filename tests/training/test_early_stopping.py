import pytest

from patternsmith.training.early_stopping import EarlyStopping


def test_first_validation_loss_is_improvement():
    stopper = EarlyStopping(patience=3)

    assert stopper.step(2.0) is False
    assert stopper.best_loss == 2.0
    assert stopper.counter == 0


def test_improvement_resets_counter():
    stopper = EarlyStopping(patience=3)

    stopper.step(2.0)
    stopper.step(2.1)

    assert stopper.counter == 1

    stopper.step(1.8)

    assert stopper.best_loss == 1.8
    assert stopper.counter == 0


def test_stops_after_patience_is_exhausted():
    stopper = EarlyStopping(patience=3)

    stopper.step(2.0)

    assert stopper.step(2.1) is False
    assert stopper.step(2.2) is False
    assert stopper.step(2.3) is True
    assert stopper.should_stop is True


def test_min_delta_controls_improvement():
    stopper = EarlyStopping(patience=2, min_delta=0.1)

    stopper.step(2.0)

    assert stopper.step(1.95) is False
    assert stopper.best_loss == 2.0
    assert stopper.counter == 1

    assert stopper.step(1.8) is False
    assert stopper.best_loss == 1.8
    assert stopper.counter == 0


def test_non_finite_loss_counts_as_no_improvement():
    stopper = EarlyStopping(patience=2)

    stopper.step(2.0)

    assert stopper.step(float("nan")) is False
    assert stopper.step(float("inf")) is True


@pytest.mark.parametrize(
    "patience,min_delta",
    [
        (0, 0.0),
        (-1, 0.0),
        (3, -0.1),
        (3, float("inf")),
        (3, float("nan")),
    ],
)
def test_rejects_invalid_configuration(patience, min_delta):
    with pytest.raises(ValueError):
        EarlyStopping(patience=patience, min_delta=min_delta)


def test_state_dict_restores_stopping_progress():
    original = EarlyStopping(patience=3, min_delta=0.01)

    original.step(2.0)
    original.step(2.1)

    restored = EarlyStopping(patience=1)
    restored.load_state_dict(original.state_dict())

    assert restored.patience == 3
    assert restored.min_delta == 0.01
    assert restored.best_loss == 2.0
    assert restored.counter == 1


def test_load_state_dict_rejects_missing_keys():
    stopper = EarlyStopping()

    with pytest.raises(ValueError, match="Missing keys"):
        stopper.load_state_dict({"patience": 3})
