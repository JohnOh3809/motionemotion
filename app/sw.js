// MotionEmotion service worker — makes the app installable and work offline
// after the first visit (pose model + wasm are cached on first use).
const CACHE = "motionemotion-v1";
const SHELL = ["./", "index.html", "record.html", "manifest.webmanifest",
               "icons/icon.svg", "icons/icon-192.png", "icons/icon-512.png"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

// Same-origin + CDN assets: cache-first with background refresh of the shell.
// The .onnx model is fetched with cache "no-store" by the page, so a newly
// trained model is always picked up.
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET") return;
  if (url.pathname.endsWith(".onnx") || url.pathname.endsWith("motionemotion.json")) return;
  const cacheable = url.origin === location.origin
    || url.hostname === "cdn.jsdelivr.net"
    || url.hostname === "storage.googleapis.com"
    || url.hostname.endsWith("gstatic.com")
    || url.hostname === "fonts.googleapis.com";
  if (!cacheable) return;

  e.respondWith(
    caches.match(e.request).then((hit) => {
      const fetched = fetch(e.request).then((res) => {
        if (res.ok) caches.open(CACHE).then((c) => c.put(e.request, res.clone())).catch(() => {});
        return res.clone();
      }).catch(() => hit);
      return hit || fetched;
    })
  );
});
