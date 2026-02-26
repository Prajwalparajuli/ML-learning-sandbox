# ML Sandbox

Interactive classical machine learning sandbox built with React, TypeScript, Vite, Zustand, Tailwind CSS, and Plotly.

The app is focused on learning through direct manipulation:
- Change model and hyperparameters
- Observe decision boundaries or fit behavior
- Validate with metrics and diagnostics
- Use guided presets and mission flows

Deep learning content is intentionally excluded.

## Repository Layout

```
ML Sandbox/
  app/                         # Frontend application (Vite)
    src/
    package.json
    vite.config.ts
  .github/workflows/
    deploy-pages.yml           # GitHub Pages deployment workflow
  README.md
```

Notes:
- `docs/` is intentionally ignored and not part of deployment.
- Deploy output is generated in `app/dist`.

## Tech Stack

- React 19
- TypeScript
- Vite 7
- Zustand (state management)
- Tailwind CSS
- Plotly (`react-plotly.js`)
- Radix UI primitives + utility components

## Local Development

Prerequisites:
- Node.js 20+
- npm 10+

Install and run:

```bash
cd app
npm ci
npm run dev
```

Build locally:

```bash
cd app
npm run build
```

## GitHub Pages Deployment

This repo is configured for GitHub Actions based deployment to GitHub Pages.

Workflow file:
- `.github/workflows/deploy-pages.yml`

How it works:
1. Trigger on push to `main` (or manual dispatch).
2. Install dependencies in `app/`.
3. Run `npm run build` in `app/`.
4. Upload `app/dist` as Pages artifact.
5. Deploy with `actions/deploy-pages`.

### Required repository settings

In your GitHub repository:
1. Go to `Settings -> Pages`.
2. Under `Build and deployment`, set `Source` to `GitHub Actions`.
3. Ensure default branch is `main` (or update workflow branch filter).

No extra secrets are required for standard Pages deployment.

## Base Path and Routing

`app/vite.config.ts` uses:

```ts
base: './'
```

This keeps built asset paths relative, which works well for project Pages deployments.

## Quality and Build Notes

- Type checking and build are run together via:
  - `npm run build` -> `tsc -b && vite build`
- Bundle warning threshold is configured to avoid false alarms from Plotly-heavy builds.

## Recommended Git Initialization Flow

From repo root:

```bash
git init
git add .
git commit -m "Initial commit: ML Sandbox with GitHub Pages deployment"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

## Contribution Guidelines

When making changes:
- Keep UI wiring intact.
- Keep classical ML focus; do not introduce deep-learning curriculum scope.
- Avoid introducing passive cards with no action in focus mode.
- Verify with `npm run build` before pushing.

