function startVideo() {
  const video = document.getElementById('video-stream');
  video.src = "/video_feed";
}

function stopVideo() {
  const video = document.getElementById('video-stream');
  video.src = "";
}
