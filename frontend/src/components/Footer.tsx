
export default function Footer() {
  return (
    <footer className="border-t border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-950 py-8">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-zinc-500">
        <p>© QuickMachine — Automated Machine Learning, Hyperparameter Tuning & Stacking Platform.</p>
        <div className="flex gap-4">
          <span>FastAPI + Next.js Migration</span>
          <span className="text-zinc-300 dark:text-zinc-700">|</span>
          <span>Optuna & Keras-Tuner Powered</span>
        </div>
      </div>
    </footer>
  );
}
