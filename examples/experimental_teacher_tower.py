"""A static student–teacher tower with branch-local teacher gradient cuts."""

from cutilereduce.experimental import (
    Add,
    FoldMaps,
    FoldOperator,
    FoldSpec,
    LogSumExp,
    LogWeightedSum,
    Product,
    StructuredValue,
    TensorSpec,
)


def make_teacher_tower(num_predictors: int = 3) -> FoldOperator:
    if num_predictors < 1:
        raise ValueError("a tower needs at least one predictor")
    algebra = Product(
        **{
            **{f"normalizer_{i}": LogSumExp() for i in range(num_predictors)},
            **{
                f"expectation_{i}": LogWeightedSum(())
                for i in range(num_predictors - 1)
            },
            "truth": Add(),
        }
    )
    spec = FoldSpec(
        inputs={
            **{
                f"logits_{i}": TensorSpec(("b", "r"), "float32")
                for i in range(num_predictors)
            },
            "target": TensorSpec(("b",), "int64"),
        },
        batch_axes=("b",),
        fold_axes=("r",),
    )

    def to_factor(ctx, inputs):
        logits = [inputs[f"logits_{i}"] for i in range(num_predictors)]
        components = {
            f"normalizer_{i}": StructuredValue({"value": value})
            for i, value in enumerate(logits)
        }
        components.update(
            {
                f"expectation_{i}": StructuredValue(
                    {
                        "log_weight": ctx.stop_gradient(logits[i + 1]),
                        "value": logits[i],
                    }
                )
                for i in range(num_predictors - 1)
            }
        )
        components["truth"] = StructuredValue(
            {
                "value": ctx.where(
                    ctx.compare("eq", ctx.axis_index("r"), inputs["target"]),
                    logits[-1],
                    ctx.full(logits[-1].axes, 0.0),
                )
            }
        )
        return algebra.value(**components)

    def to_output(ctx, semantic):
        del ctx
        losses = {
            f"loss_{i}": semantic[f"normalizer_{i}"]["value"]
            - semantic[f"expectation_{i}" if i < num_predictors - 1 else "truth"][
                "value"
            ]
            for i in range(num_predictors)
        }
        return StructuredValue({**losses, "total": sum(losses.values())})

    return FoldOperator(
        spec, algebra, maps=FoldMaps(to_factor=to_factor, to_output=to_output)
    )


if __name__ == "__main__":
    print(make_teacher_tower().lower().map_fold.format())
