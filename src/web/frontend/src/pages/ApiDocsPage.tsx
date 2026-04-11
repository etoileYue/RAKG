import { getApiDocsUrl } from '../api';

export function ApiDocsPage() {
  return (
    <div className="space-y-3">
      <section className="panel p-4">
        <h2 className="m-0 font-display text-lg font-semibold">API</h2>
        <p className="m-0 mt-1 text-sm text-app-muted">内嵌 FastAPI Swagger 文档（/docs）</p>
      </section>

      <section className="panel overflow-hidden p-0">
        <iframe
          title="RAKG API Docs"
          src={getApiDocsUrl()}
          className="h-[calc(100vh-220px)] min-h-[560px] w-full border-0"
          loading="lazy"
        />
      </section>
    </div>
  );
}
