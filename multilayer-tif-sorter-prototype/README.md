# Multi-layer TIFF Sorter Prototype

기존 프로젝트와 분리된 새 데스크톱 애플리케이션입니다. 배열 index 1의 세포핵 오브젝트와 index 5/6의 오브젝트가 겹치지 않는 멀티레이어 TIFF를 선별합니다.

## 실행

실행 환경이 프로젝트 내부에 구성되어 있습니다. Windows에서 `run.bat`을 더블클릭합니다.

환경을 다시 구성해야 할 때는 `setup.bat`을 실행합니다.

수동 실행:

```powershell
.venv-app\Scripts\python.exe -m pip install -r requirements.txt
.venv-app\Scripts\python.exe app.py
```

## 분석 기준

- 채널 번호는 Python 배열처럼 0부터 시작하며 `index 1`, `index 5`, `index 6`을 사용합니다.
- 각 채널에 Otsu 자동 임계값을 적용해 오브젝트 마스크를 만듭니다.
- 설정한 크기보다 작은 연결 영역은 노이즈로 제거합니다.
- index 1 마스크 중 index 5/6 마스크와 겹친 픽셀 비율을 계산합니다.
- index 5/6은 형태학적 opening으로 완만한 로컬 배경을 뺀 후 밝은 오브젝트를 검출합니다. UI에서 배경 제거 반경(px)을 조절할 수 있으며 0은 비활성화입니다.
- 기본값 `허용 겹침률 0%`에서는 한 픽셀도 겹치지 않아야 선택됩니다.
- 선택된 TIFF는 출력 폴더로 복사되고 전체 판정은 `analysis_report.csv`에 기록됩니다.
- 원본 TIFF는 변경하지 않습니다.

`대표 이미지 미리보기` 버튼을 누르면 전체 선별 전에 TIFF 한 장을 선택하여 현재 설정 기준의 RGB 합성 영상, 예상 판정, 겹침률을 확인할 수 있습니다.

추가 T/Z 축이 있는 데이터는 밝은 형광 오브젝트를 놓치지 않도록 최대강도 투영(maximum-intensity projection) 후 분석합니다. 미리보기는 원본 합성, 실제 검출 마스크, 겹침 강조 화면 순서이며 겹친 픽셀은 노란색으로 표시됩니다.
