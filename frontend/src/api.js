/**
 * EchoTrace Frontend API Client
 * Connects to the FastAPI backend without exposing any credentials or secrets.
 */

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/+$/, '');

export { API_BASE_URL };

/**
 * Health check to verify backend connectivity.
 */
export async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (!res.ok) return false;
    const data = await res.json();
    return data.status === 'healthy';
  } catch (err) {
    console.error('Backend health check error:', err);
    return false;
  }
}

/**
 * Uploads an audio or video file to trigger the ingestion pipeline.
 */
export async function uploadMediaFile(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE_URL}/media/upload?async_process=true`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    let errDetail = 'Failed to upload media';
    try {
      const errJson = await res.json();
      errDetail = errJson.detail || errDetail;
    } catch (_) {}
    throw new Error(errDetail);
  }

  return await res.json();
}

/**
 * Ingests media from a direct audio/video URL or podcast RSS feed.
 */
export async function ingestMediaUrl(url, requestedEpisode = null) {
  const res = await fetch(`${API_BASE_URL}/media/url?async_process=true`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url, requested_episode: requestedEpisode }),
  });

  if (!res.ok) {
    let errDetail = 'Failed to ingest URL';
    try {
      const errJson = await res.json();
      errDetail = errJson.detail || errDetail;
    } catch (_) {}
    throw new Error(errDetail);
  }

  return await res.json();
}

/**
 * Lists all uploaded recordings and their statuses.
 */
export async function listRecordings() {
  const res = await fetch(`${API_BASE_URL}/media/recordings`);
  if (!res.ok) {
    throw new Error('Failed to fetch recordings list');
  }
  return await res.json();
}

/**
 * Polls the current processing status of a recording.
 */
export async function getRecordingStatus(recordingId) {
  const res = await fetch(`${API_BASE_URL}/media/${encodeURIComponent(recordingId)}/status`);
  if (!res.ok) {
    throw new Error(`Failed to fetch status for recording ${recordingId}`);
  }
  return await res.json();
}

/**
 * Fetches timestamped transcript chunks for a ready recording.
 */
export async function getRecordingTranscript(recordingId) {
  const res = await fetch(`${API_BASE_URL}/media/${encodeURIComponent(recordingId)}/transcript`);
  if (!res.ok) {
    throw new Error(`Failed to fetch transcript for recording ${recordingId}`);
  }
  return await res.json();
}

/**
 * Deletes a recording, its metadata, media files, and vector chunks.
 */
export async function deleteRecording(recordingId) {
  const res = await fetch(`${API_BASE_URL}/media/${encodeURIComponent(recordingId)}`, {
    method: 'DELETE',
  });

  if (!res.ok) {
    let errDetail = 'Failed to delete recording';
    try {
      const errJson = await res.json();
      errDetail = errJson.detail || errDetail;
    } catch (_) {}
    throw new Error(errDetail);
  }

  return await res.json();
}

/**
 * Returns the direct media streaming URL for audio/video playback with HTTP Range support.
 */
export function getMediaStreamUrl(recordingId) {
  return `${API_BASE_URL}/media/stream/${encodeURIComponent(recordingId)}`;
}

/**
 * Queries indexed transcripts using the evidence-grounded RAG endpoint.
 */
export async function askRAGQuestion(query, recordingId = null, topK = 5) {
  const payload = {
    query,
    top_k: topK,
  };
  if (recordingId) {
    payload.recording_id = recordingId;
  }

  const res = await fetch(`${API_BASE_URL}/api/rag/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    let errDetail = 'Failed to generate answer';
    try {
      const errJson = await res.json();
      errDetail = errJson.detail || errDetail;
    } catch (_) {}
    throw new Error(errDetail);
  }

  return await res.json();
}
