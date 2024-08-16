from typing import List

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerBase

from grouse.dtos import (
    EvaluationSample,
    Failed,
    MetaTestCaseResult,
)

VALUE_COLORS = {1: "tab:blue", 0: "tab:red", 2: "tab:orange"}
VALUE_LABELS = {1: "Test passed", 0: "Test failed", 2: "Output wrong format"}
CMAP = plt.cm.colors.ListedColormap([VALUE_COLORS[0], VALUE_COLORS[1], VALUE_COLORS[2]])
NORM = mcolors.BoundaryNorm(boundaries=[0, 1, 2, 3], ncolors=3)


def add_custom_legend(
    hatch_legend: bool = True,
    value_colors=VALUE_COLORS,
    value_labels=VALUE_LABELS,
    fig=None,
    ax=None,
    *args,
    **kwargs,
):
    if fig is None and ax is None:
        raise ValueError("Both fig and ax can't be None")
    if fig is not None and ax is not None:
        raise ValueError("Both fig and ax can't be different than None")
    fig_or_ax = fig or ax

    patch_size = 20
    list_patches = []
    for key in value_colors.keys():
        list_patches.append(
            patches.Rectangle(
                (0, 0),
                patch_size,
                patch_size,
                color=value_colors[key],
                label=value_labels[key],
            )
        )
    if hatch_legend:
        list_patches.append(
            patches.Rectangle(
                (0, 0),
                patch_size,
                patch_size,
                facecolor="white",
                label="Theoretically won't happen",
                hatch="/////",
            )
        )
    fig_or_ax.legend(
        *args, handles=list_patches, handlelength=1, handleheight=1, **kwargs
    )


def plot_matrix(
    values, row_names, title, ax, hatch_rows, show_yticks=False, show_xlabel=False
):
    matrix = []
    for row_index in range(16):
        column = []
        for column_index in range(9):
            value = values[column_index * 16 + row_index]
            if np.isnan(value):
                column.append(2)
            else:
                column.append(value)
        matrix.append(column)
    matrix = np.array(matrix)

    # Adds white lines between squares
    for k in range(matrix.shape[0] - 1):
        ax.axhline(k + 0.5, color="white", linewidth=2)
    for k in range(matrix.shape[1] - 1):
        ax.axvline(k + 0.5, color="white", linewidth=2)

    ax.imshow(matrix, cmap=CMAP, norm=NORM, interpolation="nearest")
    for row in hatch_rows:
        for column in range(matrix.shape[1]):
            ax.add_patch(
                patches.Rectangle(
                    (column - 0.5, row - 0.5),
                    1,
                    1,
                    hatch="///",
                    fill=False,
                    snap=False,
                    linewidth=0,
                )
            )

    # display all ticks on the x-axis
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(np.arange(matrix.shape[1]))

    # display all ticks on the y-axis
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(np.arange(matrix.shape[0]))

    xticklabels = ax.get_xticklabels()
    xtick_positions = ax.get_xticks()

    ax.set_xticklabels(
        list(range(len(xticklabels))), rotation=0, ha="center", fontsize=10
    )

    # Plot a circle around each xtick label
    for pos in xtick_positions:
        circle = plt.Circle((pos, 16.3), 0.4, color="black", fill=False, clip_on=False)
        ax.add_artist(circle)

    # Change the xticks and yticks labels
    ax.set_xticklabels(list(range(1, matrix.shape[1] + 1)), fontsize=14)

    if show_xlabel:
        ax.set_xlabel("Test samples", labelpad=10, fontsize=16)

    if show_yticks:
        ax.set_yticklabels(row_names, fontsize=14)
        ax.set_ylabel("Test type", labelpad=10, fontsize=16)
    else:
        ax.set_yticklabels(list(range(1, matrix.shape[0] + 1)), fontsize=14)
    ax.set_title(title, fontsize=18)


def build_circle_legend(letter_to_label_dict):
    handler_map = {}
    fake_objects = []
    labels = []
    for letter, label in letter_to_label_dict.items():
        HandlerBase  # to avoid having the import removed by isort or something

        exec(
            f"""class myclass_{letter}:
        pass"""
        )
        exec(f"fake_objects.append(myclass_{letter}())")

        exec(
            f"""class LetterLegendHandler{letter}(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        text = "{letter}"
        text_artist = plt.Text(
            xdescent + 0.5 * width, ydescent + 0.35 * height, text, fontsize=fontsize, ha="center", va="center"
        )
        circle_artist = plt.Circle((xdescent + 0.5 * width, ydescent + 0.4 * height), width * 0.32, fill=False)
        return [text_artist, circle_artist]"""
        )

        exec(f"handler_map[myclass_{letter}] =  LetterLegendHandler{letter}()")
        labels.append(label)

    return fake_objects, labels, handler_map


def process_value(value):
    if isinstance(value, Failed):
        return 2
    else:
        return int(value)


def plot_matrices(
    evaluation_samples: List[EvaluationSample],
    meta_evaluations: List[MetaTestCaseResult],
) -> None:
    questions_seen = set()
    unique_questions = []
    test_types_seen = set()
    unique_test_types = []

    for evaluation_sample in evaluation_samples:
        if evaluation_sample.input not in questions_seen:
            unique_questions.append(evaluation_sample.input)
            questions_seen.add(evaluation_sample.input)
        test_type = evaluation_sample.metadata.get("test_type", "")
        if test_type not in test_types_seen:
            unique_test_types.append(test_type)
            test_types_seen.add(test_type)

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(40, 10))
    fig.subplots_adjust(bottom=0.5, left=0.2)

    answer_relevancy_values = [
        process_value(e.answer_relevancy) for e in meta_evaluations
    ]
    completeness_values = [process_value(e.completeness) for e in meta_evaluations]
    faithfulness_values = [process_value(e.faithfulness) for e in meta_evaluations]
    usefulness_values = [process_value(e.usefulness) for e in meta_evaluations]

    plot_matrix(
        answer_relevancy_values,
        unique_test_types,
        "Answer Relevancy",
        axes[0],
        [],
        show_yticks=True,
        show_xlabel=True,
    )
    plot_matrix(
        completeness_values,
        unique_test_types,
        "Completeness",
        axes[1],
        [],
        show_yticks=False,
        show_xlabel=True,
    )
    plot_matrix(
        usefulness_values,
        unique_test_types,
        "Usefulness",
        axes[2],
        [0, 3, 5, 7, 8, 9, 13, 14, 15],
        show_yticks=False,
        show_xlabel=True,
    )
    plot_matrix(
        faithfulness_values,
        unique_test_types,
        "Faithfulness",
        axes[3],
        [1, 4, 10],
        show_yticks=False,
        show_xlabel=True,
    )

    bbox_to_anchor = (0, -0.7)
    custom_legend_loc = "lower left"
    value_labels = {0: "Test failed", 1: "Test passed", 2: "Output Wrong Format"}
    add_custom_legend(
        hatch_legend=True,
        value_colors=VALUE_COLORS,
        value_labels=value_labels,
        ax=axes[2],
        bbox_to_anchor=bbox_to_anchor,
        loc=custom_legend_loc,
        fontsize=18,
    )
    int_to_label_dict = {
        str(i + 1): question for (i, question) in enumerate(unique_questions)
    }
    test_questions_legend_bbox_to_anchor = (0.3, 0.3)
    fake_objects_circle, labels_circle, handler_map_circle = build_circle_legend(
        int_to_label_dict
    )
    fig.legend(
        handles=fake_objects_circle,
        labels=labels_circle,
        handler_map=handler_map_circle,
        bbox_to_anchor=test_questions_legend_bbox_to_anchor,
        loc="center",
        title="Test questions",
        title_fontproperties={"weight": "bold"},
    )
    # return fig, axes
    plt.show()
