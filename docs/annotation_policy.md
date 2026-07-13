# Person annotation policy

This policy defines the target object for VisionTrack domain data. It must be
applied consistently to domain positives, hard negatives, validation data, and
the isolated final test set.

## Positive person

Draw one tight person bounding box when a real person is unambiguously
recognizable, including:

- a fully visible person;
- a partially visible or strongly occluded person with enough visible silhouette
  to identify the object reliably;
- a distant person while the annotation remains reliable.

Do not invent a box when the visible evidence is ambiguous. Mark the sample for
review instead.

## Distractors and hard negatives

Do not draw a person box around:

- an isolated hand, arm, leg, or body fragment without enough person silhouette;
- a person shown on a monitor, television, phone, poster, or photograph;
- a reflection, because VisionTrack counting and tracking must ignore reflected
  people;
- person-like furniture, clothing, lamps, chairs, or other background objects.

Images containing only these distractors are hard-negative images. Ultralytics
accepts such an image without a label file; an empty label file is also
representable, but the project convention is to omit it.

## Mixed images

An image is not globally positive or negative. Annotate every real person that
meets the positive rule and leave each distractor unboxed. For example, in a
frame containing a real person and that person's reflection, label the real
person only. Never convert this mixed frame into an empty negative.

## Quality rules

- Boxes should cover the visible person extent without including unrelated
  people or large background regions.
- Preserve difficult positives: partial, distant, crowded, and occluded people
  are not removed merely because they are hard.
- Record ambiguous samples as `pending` in the domain manifest and exclude them
  from training/evaluation until reviewed.
- Review contact sheets for missed real people, boxes on distractors, loose or
  truncated boxes, dense crowds, reflections, screens, lighting extremes, and
  indoor/outdoor balance.
- A label-free image is valid only after confirming that it contains no object
  considered positive by this policy.

## Change control

The policy is part of the dataset version. If product behavior changes—for
example, reflections become countable—the affected data must be relabeled and
the dataset version must change. Do not mix policies in one evaluation split.
