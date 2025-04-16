# 📚 Docs guide

How to work with documentation

## Setup

To work with the documentation, you'll need to install the documentation dependencies:

```bash
# Install documentation dependencies
uv sync --group docs
```

## Development

### Local Development Server

To preview the documentation locally while editing:

```bash
mkdocs serve
```

This will start a development server at [http://127.0.0.1:8000](http://127.0.0.1:8000) that automatically refreshes when you make changes to the documentation files.

### Adding New Pages

To add a new page to the documentation create a Markdown file in the appropriate directory under `docs/`

### Adding Images and Assets

Store images and other assets in the `docs/assets/` directory:

```
docs/assets/
├── images/            # For screenshots and diagrams
├── diagrams/          # For architectural diagrams
└── examples/          # For example files
```

Reference images in your Markdown using relative paths:

```markdown
![Database Schema](../assets/images/database-schema.png)
```

## Building

To build a static version of the documentation:

```bash
mkdocs build
```

## Deployment

The documentation is automatically deployed when changes are pushed to the main branch. You can also manually deploy:

```bash
# Deploy to GitHub Pages
mkdocs gh-deploy
```
