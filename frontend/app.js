const TARGET_CAPTURE_FRAMES = 60;
const FALLBACK_FRAME_RATE = 30;
const PREPARATION_SECONDS = 2;
const camera = document.querySelector("#camera");
const captureButton = document.querySelector("#capture-button");
const status = document.querySelector("#status");
const result = document.querySelector("#result");
const countdown = document.querySelector("#countdown");
const audioResult = document.querySelector("#audio-result");

let cameraStream = null;
let busy = false;
let audioUrl = null;
let synthesizerPromise = null;

function setStatus(message, state = "") {
  status.textContent = message;
  status.dataset.state = state;
}

function wait(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
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

async function checkBackend() {
  let response;
  try {
    response = await fetch("/health");
  } catch (error) {
    throw new Error(
      "Backend inaccessible. Lancez FastAPI dans backend puis ouvrez http://127.0.0.1:8000.",
      { cause: error },
    );
  }

  if (!response.ok) {
    throw new Error(
      "Backend indisponible. Ouvrez l’application via http://127.0.0.1:8000, pas depuis un serveur statique.",
    );
  }

  const health = await response.json();
  if (health.status !== "ok" || health.modele !== "model_gru.tflite") {
    throw new Error("Le backend ou le modèle de reconnaissance n’est pas prêt.");
  }
}

async function prepareCapture() {
  for (let seconds = PREPARATION_SECONDS; seconds > 0; seconds -= 1) {
    countdown.textContent = String(seconds);
    countdown.hidden = false;
    setStatus(`Préparez-vous : ${seconds}`, "busy");
    await wait(1000);
  }
  countdown.hidden = true;
  countdown.textContent = "";
}

function captureDurationMs() {
  const frameRate = cameraStream.getVideoTracks()[0]?.getSettings().frameRate;
  const effectiveFrameRate =
    Number.isFinite(frameRate) && frameRate > 0 ? frameRate : FALLBACK_FRAME_RATE;
  return (TARGET_CAPTURE_FRAMES / effectiveFrameRate) * 1000;
}

function recordClip() {
  return new Promise((resolve, reject) => {
    if (typeof MediaRecorder === "undefined") {
      reject(new Error("L’enregistrement vidéo est indisponible dans ce navigateur."));
      return;
    }
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
    setStatus("Capture…", "busy");
    window.setTimeout(() => {
      if (recorder.state === "recording") {
        recorder.stop();
      }
    }, captureDurationMs());
  });
}

function makeWav(audio, sampleRate) {
  if (!(audio instanceof Float32Array) || audio.length === 0 || !Number.isFinite(sampleRate)) {
    throw new Error("Le modèle audio a renvoyé un signal invalide.");
  }

  const bytesPerSample = 2;
  const buffer = new ArrayBuffer(44 + audio.length * bytesPerSample);
  const view = new DataView(buffer);
  const writeText = (offset, text) => {
    for (let index = 0; index < text.length; index += 1) {
      view.setUint8(offset + index, text.charCodeAt(index));
    }
  };

  writeText(0, "RIFF");
  view.setUint32(4, 36 + audio.length * bytesPerSample, true);
  writeText(8, "WAVE");
  writeText(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, bytesPerSample * 8, true);
  writeText(36, "data");
  view.setUint32(40, audio.length * bytesPerSample, true);

  for (let index = 0; index < audio.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, audio[index]));
    view.setInt16(44 + index * bytesPerSample, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

async function synthesize(text) {
  if (!synthesizerPromise) {
    synthesizerPromise = import("https://cdn.jsdelivr.net/npm/@xenova/transformers")
      .then(({ pipeline, env }) => {
        env.backends.onnx.wasm.numThreads = 1;
        setStatus("Chargement du modèle audio…", "busy");
        return pipeline("text-to-speech", "elloza/mms-tts-mlg-onnx", { quantized: true });
      })
      .catch((error) => {
        synthesizerPromise = null;
        throw new Error(
          "Impossible de charger le modèle audio malgache. Vérifiez la connexion Internet et réessayez.",
          { cause: error },
        );
      });
  }

  const synthesizer = await synthesizerPromise;
  setStatus("Génération audio…", "busy");
  const speech = await synthesizer(text);
  if (!speech || !Number.isFinite(speech.sampling_rate)) {
    throw new Error("Le modèle audio n’a pas produit de résultat exploitable.");
  }

  if (audioUrl) {
    URL.revokeObjectURL(audioUrl);
  }
  audioUrl = URL.createObjectURL(makeWav(speech.audio, speech.sampling_rate));
  audioResult.src = audioUrl;
  audioResult.hidden = false;

  try {
    await audioResult.play();
    return "Résultat prêt";
  } catch {
    return "Résultat prêt — appuyez sur lecture";
  }
}

async function translateClip(clip) {
  const formData = new FormData();
  formData.append("file", clip, "capture.webm");

  let response;
  try {
    response = await fetch("/predire", {
      method: "POST",
      body: formData,
    });
  } catch (error) {
    throw new Error(
      "Connexion au backend perdue pendant l’analyse. Vérifiez que FastAPI fonctionne sur le port 8000.",
      { cause: error },
    );
  }

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
  result.textContent = "";
  result.hidden = true;
  audioResult.pause();
  audioResult.removeAttribute("src");
  audioResult.hidden = true;
  if (audioUrl) {
    URL.revokeObjectURL(audioUrl);
    audioUrl = null;
  }
  setStatus("");

  try {
    await checkBackend();
    await openCamera();
    await prepareCapture();
    const clip = await recordClip();
    setStatus("Analyse…", "busy");
    const prediction = await translateClip(clip);
    result.textContent = prediction.signe;
    result.hidden = false;
    setStatus("Audio…", "busy");
    setStatus(await synthesize(prediction.signe));
  } catch (error) {
    console.error("Échec de la capture ou de la traduction :", error);
    setStatus(error instanceof Error ? error.message : "Une erreur est survenue.", "error");
  } finally {
    countdown.hidden = true;
    countdown.textContent = "";
    busy = false;
    captureButton.disabled = false;
    captureButton.textContent = "Capturer";
  }
});

window.addEventListener("beforeunload", () => {
  cameraStream?.getTracks().forEach((track) => track.stop());
  if (audioUrl) {
    URL.revokeObjectURL(audioUrl);
  }
});
