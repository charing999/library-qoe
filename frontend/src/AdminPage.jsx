// frontend/src/pages/AdminPage.jsx

import React, { useEffect, useState } from "react";
import axios from "axios";

const API_BASE = "http://localhost:8000";

export default function AdminPage() {
  const [dashboard, setDashboard] = useState(null);
  const [selectedAp, setSelectedAp] = useState(null);
  const [currentFloor, setCurrentFloor] = useState("1F"); // 기본층

  // 층별 대시보드 데이터 로드
  useEffect(() => {
    setDashboard(null);

    axios
      .get(`${API_BASE}/api/dashboard?floor=${currentFloor}`)
      .then((res) => {
        console.log("✅ 대시보드 데이터:", res.data);
        setDashboard(res.data);
        setSelectedAp(null);
      })
      .catch((err) => {
        console.error("❌ 대시보드 로딩 실패:", err);
      });
  }, [currentFloor]);

  // AP 클릭 → 상세 예측
  const handleApClick = (apId) => {
    axios
      .get(`${API_BASE}/api/predict/${apId}`)
      .then((res) => {
        console.log("✅ 상세 예측:", res.data);
        setSelectedAp(res.data);
      })
      .catch((err) => {
        console.error("❌ 예측 로딩 실패:", err);
      });
  };

  if (!dashboard) {
    return (
      <div className="p-10 text-xl font-bold">
        📡 데이터를 불러오는 중...
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-gray-100">
      {/* 🔹 왼쪽: 지도 + 요약 */}
      <div className="flex-1 p-6 flex flex-col">
        {/* 헤더 + 층 선택 */}
        <div className="flex justify-between items-center mb-4">
          <h1 className="text-3xl font-bold text-gray-800">
            관리자 대시보드 <span className="text-blue-600">({currentFloor})</span>
          </h1>

          <select
            value={currentFloor}
            onChange={(e) => setCurrentFloor(e.target.value)}
            className="p-2 border-2 border-blue-500 rounded-lg text-lg font-bold bg-white shadow-sm cursor-pointer hover:bg-blue-50"
          >
            <option value="B2">B2 (지하 2층)</option>
            <option value="B1">B1 (지하 1층)</option>
            <option value="1F">1F (1층)</option>
            <option value="2F">2F (2층)</option>
          </select>
        </div>

        {/* 🔹 지도 + AP 마커 */}
        <div className="flex-1 bg-white border-2 border-gray-300 rounded-xl shadow-md overflow-hidden">
          {/* ⚠️ 이 div가 AP 버튼들의 기준이 되는 컨테이너 */}
          <div
            className="w-full h-[480px]"
            style={{ position: "relative" }} // ← 인라인로 확실히 relative
          >
            {/* 평면도 이미지 */}
            <img
              src={`/maps/${currentFloor}.png`}
              alt="Floor Plan"
              className="w-full h-full object-contain cursor-crosshair"
              onClick={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                const x = ((e.clientX - rect.left) / rect.width) * 100;
                const y = ((e.clientY - rect.top) / rect.height) * 100;

                const msg = `"${selectedAp ? selectedAp.ap_id : "AP_ID_HERE"}": (${x.toFixed(
                  1
                )}, ${y.toFixed(1)}),`;
                console.log(msg);
                alert(
                  `이 위치의 좌표를 backend FIXED_POSITIONS에 붙여넣으세요:\n\n${msg}`
                );
              }}
            />

            {/* 🔥 지도 위 AP 마커들 */}
            {dashboard.aps.map((ap) => (
              <button
                key={ap.id}
                className={`w-8 h-8 rounded-full border-2 border-white shadow-md flex items-center justify-center font-bold text-[10px] text-white transition-transform hover:scale-125
                  ${
                    ap.status === "Good"
                      ? "bg-green-500"
                      : ap.status === "Moderate"
                      ? "bg-yellow-400"
                      : "bg-red-500"
                  }`}
                style={{
                  position: "absolute",        // ← 인라인 absolute
                  left: `${ap.x}%`,
                  top: `${ap.y}%`,
                  transform: "translate(-50%, -50%)",
                }}
                onClick={(e) => {
                  e.stopPropagation();
                  handleApClick(ap.id);
                }}
                title={`${ap.id} (${ap.status})`}
              >
                AP
              </button>
            ))}

            {/* 데이터 없을 때 */}
            {dashboard.aps.length === 0 && (
              <div
                className="flex items-center justify-center bg-gray-100/80 text-gray-500 font-bold"
                style={{
                  position: "absolute",
                  inset: 0,
                }}
              >
                ⚠️ 데이터가 없습니다. CSV의 'location2'와 floor 값을 확인하세요.
              </div>
            )}
          </div>
        </div>

        {/* 요약 바 */}
        <div className="mt-4 p-4 bg-white rounded-lg shadow flex justify-between">
          <span className="font-bold text-gray-600">
            총 AP 개수: {dashboard.aps.length}개
          </span>
          <span className="font-bold text-red-500">
            점검 필요: {dashboard.alert_count}개
          </span>
        </div>
      </div>

      {/* 🔹 오른쪽: 상세 패널 */}
      <div className="w-96 bg-white border-l p-6 shadow-2xl overflow-y-auto">
        <h2 className="text-2xl font-bold mb-6 border-b pb-2">상세 정보</h2>

        {selectedAp ? (
          <div className="space-y-6">
            {/* AP 이름 */}
            <div className="p-4 bg-blue-50 rounded-xl border border-blue-100">
              <p className="text-xs text-blue-500 font-bold mb-1">AP ID</p>
              <h3 className="text-lg font-bold text-blue-800 break-all leading-tight">
                {selectedAp.ap_id}
              </h3>
            </div>

            {/* 현재 / 5분 후 등급 카드 */}
            <div className="grid grid-cols-2 gap-4">
              {/* 현재 상태 */}
              <div className="bg-gray-50 p-4 rounded-xl text-center shadow-sm">
                <p className="text-xs text-gray-500 mb-1">
                  {selectedAp.current_time_text}
                </p>
                <span
                  className={`inline-block px-3 py-1 rounded-full text-sm font-bold 
                    ${
                      selectedAp.current_grade === "Good"
                        ? "bg-green-100 text-green-700"
                        : selectedAp.current_grade === "Moderate"
                        ? "bg-yellow-100 text-yellow-700"
                        : "bg-red-100 text-red-700"
                    }`}
                >
                  {selectedAp.current_grade}
                </span>
                <p className="text-xs text-gray-400 mt-2">
                  점수: {selectedAp.current_qoe}
                </p>
              </div>

              {/* 5분 뒤 예측 */}
              <div className="bg-blue-50 p-4 rounded-xl text-center border-2 border-blue-200 shadow-sm relative overflow-hidden">
                <div className="absolute top-0 left-0 w-full h-1 bg-blue-500" />
                <p className="text-xs text-blue-500 mb-1">
                  {selectedAp.future_time_text}
                </p>
                <span className="text-xl font-black text-blue-700">
                  {selectedAp.future_grade}
                </span>
                <p className="text-xs text-blue-400 mt-1">
                  예측 점수: {selectedAp.future_qoe}
                </p>
              </div>
            </div>

            {/* 수치 데이터 */}
            <div className="bg-gray-50 p-5 rounded-xl border space-y-3 text-sm text-gray-700">
              <div className="flex justify-between">
                <span>Ping (지연시간)</span>
                <span className="font-bold">
                  {selectedAp.metrics.ping_ms
                    ? selectedAp.metrics.ping_ms.toFixed(1)
                    : 0}{" "}
                  ms
                </span>
              </div>
              <div className="flex justify-between">
                <span>Loss (손실률)</span>
                <span className="font-bold">
                  {selectedAp.metrics.packet_loss_rate}%
                </span>
              </div>
              <div className="flex justify-between">
                <span>RSSI (신호강도)</span>
                <span className="font-bold">
                  {selectedAp.metrics.RSSI} dBm
                </span>
              </div>
              <div className="flex justify-between border-t pt-2 mt-2">
                <span>다운로드 속도</span>
                <span className="font-bold text-blue-600">
                  {selectedAp.metrics.download_Mbps
                    ? selectedAp.metrics.download_Mbps.toFixed(1)
                    : 0}{" "}
                  Mbps
                </span>
              </div>
            </div>
          </div>
        ) : (
          <div className="h-full flex flex-col items-center justify-center text-gray-400 opacity-60">
            <svg
              className="w-16 h-16 mb-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122"
              />
            </svg>
            <p>지도에서 AP를 선택하세요.</p>
          </div>
        )}
      </div>
    </div>
  );
}
