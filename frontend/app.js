const CAPTURE_DURATION_MS = 2000;
const camera = document.querySelector("#camera");
const captureButton = document.querySelector("#capture-button");
const status = document.querySelector("#status");
const result = document.querySelector("#result");
const recordingIndicator = document.querySelector("#recording-indicator");

let cameraStream = null;
let busy = false;

function setStatus(message, state = "") {
  status.textContent = message;
  status.dataset.state = state;
}

async function openCamera() {
  if (cameraStream?.active) {
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Caméra indisponible dans ce navigateur.");
  }

  cameraStream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: "user",
      width: { ideal: 640 },
      height: { ideal: 480 },
      frameRate: { ideal: 30, max: 30 },
    },
  });
  camera.srcObject = cameraStream;
  await camera.play();
}

function recordClip() {
  return new Promise((resolve, reject) => {
    const mimeType = ["video/webm;codecs=vp8", "video/webm"].find((type) =>
      MediaRecorder.isTypeSupported(type),
    );
    if (!mimeType) {
      reject(new Error("Enregistrement vidéo indisponible dans ce navigateur."));
      return;
    }

    const chunks = [];
    let recorder;
    try {
      recorder = new MediaRecorder(cameraStream, { mimeType });
    } catch (error) {
      reject(error);
      return;
    }

    recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) {
        chunks.push(event.data);
      }
    });
    recorder.addEventListener("error", () => {
      reject(new Error("La capture vidéo a échoué."));
    });
    recorder.addEventListener(
      "stop",
      () => {
        const clip = new Blob(chunks, { type: recorder.mimeType });
        if (clip.size === 0) {
          reject(new Error("La capture vidéo est vide."));
          return;
        }
        resolve(clip);
      },
      { once: true },
    );

    recorder.start();
    recordingIndicator.dataset.active = "true";
    setStatus("Capture…", "busy");
    window.setTimeout(() => {
      if (recorder.state === "recording") {
        recorder.stop();
      }
      recordingIndicator.dataset.active = "false";
    }, CAPTURE_DURATION_MS);
  });
}

async function translateClip(clip) {
  const formData = new FormData();
  formData.append("file", clip, "capture.webm");

  const response = await fetch("/predire", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let message = "La traduction a échoué.";
    try {
      const body = await response.json();
      if (typeof body.detail === "string") {
        message = body.detail;
      }
    } catch {
      message = `Erreur serveur (${response.status}).`;
    }
    throw new Error(message);
  }

  const prediction = await response.json();
  if (
    typeof prediction.signe !== "string" ||
    !Number.isFinite(prediction.confiance)
  ) {
    throw new Error("Réponse de traduction invalide.");
  }
  return prediction;
}

captureButton.addEventListener("click", async () => {
  if (busy) {
    return;
  }

  busy = true;
  captureButton.disabled = true;
  result.value = "";
  setStatus("");

  try {
    await openCamera();
    const clip = await recordClip();
    setStatus("Traduction…", "busy");
    const prediction = await translateClip(clip);
    result.value = prediction.signe;
    setStatus("");
  } catch (error) {
    console.error("Échec de la capture ou de la traduction :", error);
    setStatus(error instanceof Error ? error.message : "Une erreur est survenue.", "error");
  } finally {
    busy = false;
    captureButton.disabled = false;
    captureButton.textContent = "Capturer";
  }
});

window.addEventListener("beforeunload", () => {
  cameraStream?.getTracks().forEach((track) => track.stop());
});
