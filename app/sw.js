// Cache the app shell and fetched assets for subsequent offline use.
// Pose assets are downloaded from CDNs on first use.
const CACHE = "motionemotion-v1"; // bump this when the shell changes
const SHELL = ["./", "index.html", "record.html", "manifest.webmanifest",
               "icons/icon.svg", "icons/icon-192.png", "icons/icon-512.png"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  // clear caches from old versions
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

// cache-first, refresh in the background. one deliberate exception: the
// .onnx model and its json are never cached here — the page fetches them
// no-store so a freshly trained model always wins over a stale one.
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
      }).catch(() => hit); // retain the cached response if the refresh fails
      return hit || fetched;
    })
  );
});
