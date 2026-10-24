import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";

const MODEL_ID = "elloza/mms-tts-mlg-onnx";
const videoInput = document.querySelector("#video-input");
const fileName = document.querySelector("#file-name");
const videoPreview = document.querySelector("#video-preview");
const translateButton = document.querySelector("#translate-button");
const statusBar = document.querySelector(".status-bar");
const statusMessage = document.querySelector("#status-message");
const emptyResult = document.querySelector("#empty-result");
const translationResult = document.querySelector("#translation-result");
const recognizedSign = document.querySelector("#recognized-sign");
const confidence = document.querySelector("#confidence");
const audioPlayer = document.querySelector("#audio-player");
const audioState = document.querySelector("#audio-state");
const audioMessage = document.querySelector("#audio-message");

env.allowLocalModels = false;
env.backends.onnx.wasm.numThreads = 1;

let selectedVideo = null;
let previewUrl = null;
let audioUrl = null;
let synthesizerPromise = null;

function setStatus(message, state = "idle") {
  statusMessage.textContent = message;
  statusBar.dataset.state = state;
}

function setAudioState(state, message) {
  audioState.textContent = state;
  audioState.dataset.state = state === "Prêt" ? "ready" : state === "Erreur" ? "error" : "loading";
  audioMessage.textContent = message;
}

function getSynthesizer() {
  if (!synthesizerPromise) {
    synthesizerPromise = pipeline("text-to-speech", MODEL_ID, {
      quantized: true,
    }).catch((error) => {
      synthesizerPromise = null;
      throw error;
    });
  }
  return synthesizerPromise;
}

function toWavBlob(audio, sampleRate) {
  if (!(audio instanceof Float32Array) || audio.length === 0) {
    throw new Error("Le modèle audio n’a pas produit de signal exploitable.");
  }
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
    throw new Error("La fréquence d’échantillonnage audio est invalide.");
  }

  const bytesPerSample = 2;
  const dataLength = audio.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataLength);
  const view = new DataView(buffer);

  function writeText(offset, text) {
    for (let index = 0; index < text.length; index += 1) {
      view.setUint8(offset + index, text.charCodeAt(index));
    }
  }

  writeText(0, "RIFF");
  view.setUint32(4, 36 + dataLength, true);
  writeText(8, "WAVE");
  writeText(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, 16, true);
  writeText(36, "data");
  view.setUint32(40, dataLength, true);

  for (let index = 0; index < audio.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, audio[index]));
    const pcm = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    view.setInt16(44 + index * bytesPerSample, pcm, true);
  }

  return new Blob([buffer], { type: "audio/wav" });
}

async function generateSpeech(text) {
  setAudioState("Chargement", "Préparation du modèle audio dans le navigateur…");
  const tts = await getSynthesizer();

  setAudioState("Génération", "Synthèse de la prononciation en cours…");
  const output = await tts(text);
  const blob = toWavBlob(output.audio, output.sampling_rate);

  if (audioUrl) {
    URL.revokeObjectURL(audioUrl);
  }
  audioUrl = URL.createObjectURL(blob);
  audioPlayer.src = audioUrl;
  audioPlayer.hidden = false;
  setAudioState("Prêt", "Audio généré localement. Appuyez sur lecture pour l’écouter.");
}

function displayPrediction(result) {
  if (
    !result ||
    typeof result.signe !== "string" ||
    !Number.isFinite(result.confiance) ||
    result.confiance < 0 ||
    result.confiance > 1
  ) {
    throw new Error("La réponse du serveur ne contient pas une prédiction valide.");
  }

  emptyResult.hidden = true;
  translationResult.hidden = false;
  recognizedSign.textContent = result.signe;
  confidence.textContent = `Confiance du modèle : ${(result.confiance * 100).toFixed(1)} %`;
}

videoInput.addEventListener("change", () => {
  selectedVideo = videoInput.files?.[0] ?? null;
  translateButton.disabled = !selectedVideo;
  fileName.textContent = selectedVideo
    ? `${selectedVideo.name} · ${(selectedVideo.size / (1024 * 1024)).toFixed(1)} Mo`
    : "Aucun fichier sélectionné";

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
  if (selectedVideo) {
    previewUrl = URL.createObjectURL(selectedVideo);
    videoPreview.src = previewUrl;
    videoPreview.hidden = false;
  } else {
    videoPreview.removeAttribute("src");
    videoPreview.hidden = true;
  }

  setStatus(
    selectedVideo ? "Vidéo prête. Lancez la traduction." : "En attente d’une vidéo.",
  );
});

translateButton.addEventListener("click", async () => {
  if (!selectedVideo) {
    setStatus("Choisissez d’abord une vidéo.", "error");
    return;
  }

  translateButton.disabled = true;
  setStatus("Analyse de la vidéo en cours…", "processing");
  audioPlayer.pause();
  audioPlayer.removeAttribute("src");
  audioPlayer.hidden = true;
  setAudioState("En attente", "L’audio sera généré après la traduction.");

  try {
    const formData = new FormData();
    formData.append("file", selectedVideo);
    const response = await fetch("/predire", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      let detail = `Erreur serveur (${response.status}).`;
      try {
        const errorBody = await response.json();
        if (typeof errorBody.detail === "string") {
          detail = errorBody.detail;
        }
      } catch {
        // Conserver le message HTTP lorsqu'aucun détail JSON n'est renvoyé.
      }
      throw new Error(detail);
    }

    const result = await response.json();
    displayPrediction(result);

    try {
      await generateSpeech(result.signe);
      setStatus("Traduction terminée. Vous pouvez écouter le résultat.", "success");
    } catch (error) {
      console.error("Échec de la synthèse audio :", error);
      setAudioState("Erreur", "La traduction est disponible, mais l’audio n’a pas pu être généré.");
      setStatus("Signe reconnu ; synthèse audio indisponible.", "error");
    }
  } catch (error) {
    console.error("Échec de la traduction :", error);
    setStatus(error instanceof Error ? error.message : "La traduction a échoué.", "error");
  } finally {
    translateButton.disabled = !selectedVideo;
  }
});

window.addEventListener("beforeunload", () => {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }
  if (audioUrl) {
    URL.revokeObjectURL(audioUrl);
  }
});
