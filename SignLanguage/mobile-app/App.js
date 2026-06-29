import { StatusBar } from "expo-status-bar";
import { CameraView, useCameraPermissions } from "expo-camera";
import { useEffect, useRef, useState } from "react";
import { Button, SafeAreaView, ScrollView, StyleSheet, Text, TextInput, View } from "react-native";

const CAPTURE_INTERVAL_MS = 700;

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef(null);
  const wsRef = useRef(null);
  const timerRef = useRef(null);
  const runningRef = useRef(false);

  const [serverUrl, setServerUrl] = useState("ws://192.168.1.100:8000/ws");
  const [status, setStatus] = useState("未连接");
  const [resultText, setResultText] = useState("—");
  const [confidence, setConfidence] = useState("-");
  const [classId, setClassId] = useState("-");
  const [top3, setTop3] = useState([]);
  const [rawMessage, setRawMessage] = useState("-");

  useEffect(() => {
    return () => {
      stopRecognition();
    };
  }, []);

  const clearLoop = () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  };

  const stopRecognition = () => {
    runningRef.current = false;
    clearLoop();
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setStatus("未连接");
  };

  const scheduleNextFrame = () => {
    clearLoop();
    timerRef.current = setTimeout(captureAndSendFrame, CAPTURE_INTERVAL_MS);
  };

  const captureAndSendFrame = async () => {
    if (!runningRef.current) return;
    if (!cameraRef.current || !wsRef.current || wsRef.current.readyState !== 1) {
      scheduleNextFrame();
      return;
    }

    try {
      const photo = await cameraRef.current.takePictureAsync({
        base64: true,
        quality: 0.5,
        skipProcessing: true,
      });
      if (photo?.base64) {
        wsRef.current.send(
          JSON.stringify({
            image: `data:image/jpeg;base64,${photo.base64}`,
          })
        );
      }
    } catch (err) {
      setStatus(`拍摄失败: ${String(err)}`);
    } finally {
      scheduleNextFrame();
    }
  };

  const startRecognition = async () => {
    if (!permission?.granted) {
      const req = await requestPermission();
      if (!req.granted) {
        setStatus("未授予相机权限");
        return;
      }
    }

    if (!serverUrl.startsWith("ws://") && !serverUrl.startsWith("wss://")) {
      setStatus("地址需以 ws:// 或 wss:// 开头");
      return;
    }

    stopRecognition();
    setStatus("连接中...");
    setResultText("—");
    setConfidence("-");
    setClassId("-");
    setTop3([]);

    const ws = new WebSocket(serverUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      runningRef.current = true;
      setStatus("已连接");
      captureAndSendFrame();
    };

    ws.onclose = () => {
      runningRef.current = false;
      clearLoop();
      setStatus("已断开");
    };

    ws.onerror = () => {
      setStatus("连接错误");
    };

    ws.onmessage = (event) => {
      setRawMessage(String(event.data ?? "-"));
      try {
        const data = JSON.parse(event.data);
        if (!data.ok) {
          setResultText("—");
          setTop3([]);
          return;
        }
        const list = Array.isArray(data.top3) ? data.top3 : [];
        const best = list.length > 0 ? list[0] : null;
        setResultText(best ? (best.text ?? "—") : (data.text ?? "—"));
        setConfidence(
          best && typeof best.confidence === "number"
            ? best.confidence.toFixed(3)
            : typeof data.confidence === "number"
              ? data.confidence.toFixed(3)
              : String(data.confidence ?? "-")
        );
        setClassId(best ? (best.class_id ?? "-") : (data.class_id ?? "-"));
        setTop3(list);
      } catch {
        setResultText("解析失败");
      }
    };
  };

  if (!permission) {
    return (
      <SafeAreaView style={styles.center}>
        <Text>正在检查相机权限...</Text>
      </SafeAreaView>
    );
  }

  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.center}>
        <Text style={styles.title}>需要相机权限</Text>
        <Button title="授权相机" onPress={requestPermission} />
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.title}>手语实时翻译（Expo Go 实验版）</Text>
        <Text style={styles.tip}>后端地址请改为你电脑局域网 IP，例如 ws://192.168.1.100:8000/ws</Text>

        <TextInput style={styles.input} value={serverUrl} onChangeText={setServerUrl} autoCapitalize="none" />

        <CameraView ref={cameraRef} style={styles.camera} facing="front" />

        <View style={styles.row}>
          <Button title="开始识别" onPress={startRecognition} />
          <View style={styles.btnGap} />
          <Button title="停止" color="#7a7a7a" onPress={stopRecognition} />
        </View>

        <Text style={styles.status}>状态: {status}</Text>
        <Text style={styles.result}>结果: {resultText}</Text>
        <Text style={styles.meta}>置信度: {confidence}</Text>
        <Text style={styles.meta}>类别ID: {classId}</Text>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Top-3 候选</Text>
          {top3.length === 0 ? (
            <Text style={styles.mono}>-</Text>
          ) : (
            top3.map((item, idx) => (
              <Text key={`${item.class_id}-${idx}`} style={styles.mono}>
                {idx + 1}. {item.text ?? "-"} [{item.class_id ?? "-"}] conf=
                {typeof item.confidence === "number" ? item.confidence.toFixed(3) : String(item.confidence ?? "-")}
              </Text>
            ))
          )}
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>后端消息</Text>
          <Text style={styles.mono}>{rawMessage}</Text>
        </View>
      </ScrollView>
      <StatusBar style="dark" />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f4efe3",
  },
  content: {
    padding: 16,
    paddingBottom: 28,
  },
  center: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
  },
  title: {
    fontSize: 22,
    fontWeight: "700",
    marginBottom: 8,
    color: "#3c3328",
  },
  tip: {
    color: "#776753",
    marginBottom: 10,
  },
  input: {
    borderWidth: 1,
    borderColor: "#d2c4a9",
    borderRadius: 10,
    backgroundColor: "#fbf7ef",
    paddingHorizontal: 12,
    paddingVertical: 10,
    marginBottom: 12,
    color: "#3c3328",
  },
  camera: {
    width: "100%",
    height: 320,
    borderRadius: 12,
    overflow: "hidden",
    marginBottom: 12,
  },
  row: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 8,
  },
  btnGap: {
    width: 10,
  },
  status: {
    color: "#645743",
    marginBottom: 6,
  },
  result: {
    fontSize: 28,
    fontWeight: "700",
    color: "#1f1a13",
    marginBottom: 6,
  },
  meta: {
    color: "#4e4334",
    marginBottom: 3,
  },
  card: {
    backgroundColor: "#fbf7ef",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#ded2bb",
    padding: 10,
    marginTop: 12,
  },
  cardTitle: {
    fontWeight: "600",
    marginBottom: 8,
    color: "#5a4d3a",
  },
  mono: {
    fontFamily: "monospace",
    color: "#3f3529",
    marginBottom: 4,
  },
});
