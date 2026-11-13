#!/usr/bin/env bash
# v20e(팀 앙상블) → 최종 제출 파일까지 사후 조정 사슬을 순서대로 다시 만든다. 결정적(난수 없음).
# 사용: bash analysis/run_postprocess_chain.sh <v20e.csv> <v10.csv> <출력 폴더>
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
V20E=$1; V10=$2; O=$(cd "$(dirname "$3")" && pwd)/$(basename "$3")
S=$O/submission_v20e_0926
$PY -I analysis/make_0926up.py "$V20E" "$V10" "$O"                         # 1. 9/26 아침·저녁 max(v20e, v10)
$PY -I analysis/adjust_0926_2019.py 0.5 "$O/submission_v20e_0926up.csv"     # 2. 9/26 2019 실측 쪽 α 0.5
$PY -I analysis/adjust_july_weekday.py 1.02 "${S}_2019a50.csv"              # 3. 7월 평일 ×1.02
$PY -I analysis/adjust_chuseok_days_2019.py 0.5 "${S}_2019a50_jul2.csv"     # 4. 9/23 낮·9/27 아침
B=${S}_2019a50_jul2_0923d0927m_a50
$PY -I analysis/adjust_ssg_0719.py "$B.csv"                                 # 5. SSG 경기 날 문학 + 7/19
$PY -I analysis/adjust_bfc_0711.py "${B}_ssg_0719.csv"                      # 6. 부천FC + 7/11 흠뻑쇼
$PY -I analysis/adjust_oct_festivals.py "${B}_ssg_0719_bfc_0711.csv"        # 7. 10월 축제 3개
$PY -I analysis/adjust_autumn_night.py "${B}_ssg_0719_bfc_0711_octfest.csv" # 8. 9~10월 밤 20~23시
$PY -I analysis/adjust_exam_days.py "${B}_ssg_0719_bfc_0711_octfest_night.csv" 0.5 3208:0.75   # 9. 학교 시험 날 (검바위 α 0.75)
$PY -I analysis/adjust_kopis.py "${B}_ssg_0719_bfc_0711_octfest_night_exam_3208a75.csv"         # 10. KOPIS 공연 2건
cp "${B}_ssg_0719_bfc_0711_octfest_night_exam_3208a75_kopis.csv" "$O/submission_final.csv"
shasum -a 256 "$O/submission_final.csv"
