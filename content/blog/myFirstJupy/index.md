---
# Documentation: https://wowchemy.com/docs/managing-content/

title: "My First Jupy"
subtitle: ""
summary: ""
authors: []
tags: []
categories: []
date: 2021-04-22T22:58:18+02:00
lastmod: 2021-04-22T22:58:18+02:00
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

```python
import plotly.express as px

df = px.data.iris()
features = ["sepal_width", "sepal_length", "petal_width", "petal_length"]

fig = px.scatter_matrix(
    df,
    dimensions=features,
    color="species"
)
fig.update_traces(diagonal_visible=False)
fig.show()
```

```html
content\blog\myFirstJupy\file.html
```

\`\`\`html
<?php insertTemplate("content\blog\myFirstJupy\file.html") ?>
\`\`\`