# Morphology-safe TIFF augmentation

세포 형태 학습용 TIF/TIFF 이미지와 마스크를 공간적으로 증강하는 도구입니다.

## 설계 원칙

- TIFF의 `X`, `Y` 축을 확인한 뒤 공간축만 변환합니다. `YXS`, `CYX`, `ZYX` 같은 배열에서 채널이나 Z축을 뒤집지 않습니다.
- 90도 회전, 좌우 반전, 정수 픽셀 이동은 보간하지 않습니다.
- 이동·작은 회전 후 빈 영역은 각 평면의 가장자리 중앙값으로 개별 채웁니다. 평균값을 계산하지 않고 실제 가장자리 픽셀 중 하나를 선택하므로 새로운 밝기값을 만들지 않습니다.
- 작은 각도 회전은 최근접 이웃만 사용하므로 입력에 없던 밝기값을 만들지 않습니다. 그러나 픽셀 복제·누락 및 경계 계단 현상은 발생하므로 기본값은 비활성화입니다.
- 이미지와 마스크에는 같은 공간 변환을 적용합니다.
- 입력의 하위 폴더 구조를 유지하고 결과를 `images/`와 `masks/`로 분리합니다.
- 출력 폴더를 입력 폴더 내부에 둘 수 있습니다. 재귀 검색 시 출력 폴더 전체를 자동 제외하여 결과를 다시 증강하지 않습니다. 단, 입력과 출력이 완전히 같은 폴더인 설정은 허용하지 않습니다.
- OME/ImageJ 축 정보, 물리적 픽셀 크기, 해상도, 채널명과 주요 TIFF 태그를 가능한 범위에서 보존합니다.
- 임시 파일에 저장하고 검증한 후 최종 파일로 교체합니다.

## 기본 변환

기본 설정은 이미지당 다음 8개 결과를 만듭니다.

- 원본
- 90도, 180도, 270도 회전
- 위 네 결과의 좌우 반전

정수 이동과 작은 각도 회전은 선택 사항이며, 기본적으로 꺼져 있습니다.

## 설치와 GUI 실행

```powershell
python -m pip install -r requirements.txt
python augment_training_images_ui.py
```

GUI에서 작은 각도 회전을 켜고 `5`를 입력하면 `-5°`, `+5°` 결과가 생성됩니다. 정수 이동에 `10`을 입력하면 위·아래·왼쪽·오른쪽 10픽셀 이동 결과가 추가됩니다.

## 명령줄 실행

```powershell
python augment_training_images.py "C:\data\images" "C:\data\augmented" --recursive
```

동일한 상대 경로와 파일명을 가진 마스크 폴더를 함께 처리할 때:

```powershell
python augment_training_images.py "C:\data\images" "C:\data\augmented" `
  --mask-input "C:\data\masks" --recursive
```

정수 10픽셀 이동을 추가할 때:

```powershell
python augment_training_images.py "C:\data\images" "C:\data\augmented" `
  --recursive --shift-pixels 10
```

최근접 이웃 방식의 ±5도 회전을 명시적으로 추가할 때:

```powershell
python augment_training_images.py "C:\data\images" "C:\data\augmented" `
  --recursive --small-angles -5 5
```

추가 옵션:

- `--overwrite`: 기존 결과 덮어쓰기
- `--no-quarter-turns`: 90/180/270도 회전 제외
- `--no-flips`: 좌우 반전 제외
- `--fill-mode border-median`: 평면별 가장자리 중앙값으로 채우기(기본값)
- `--fill-mode constant --fill-value 0`: 모든 평면을 지정한 단일 값으로 채우기

## 출력 구조

```text
augmented/
├── images/                   # 증강 이미지
│   └── 원본 하위 폴더 구조/
├── masks/                    # 마스크를 지정한 경우
│   └── 원본 하위 폴더 구조/
└── augmentation_report_*.csv
```

마스크 폴더를 지정하면 이미지와 마스크 폴더의 상대 경로가 같아야 합니다. 예를 들어 `images/class_a/cell01.tif`의 마스크는 `masks/class_a/cell01.tif`여야 합니다.

## 형태 학습 시 권장 설정

픽셀 수준 형태가 중요하면 우선 90도 회전, 좌우 반전, 작은 범위의 정수 이동과 기본 `border-median` 채우기만 사용하십시오. 다중 평면 TIFF는 각 평면의 배경값을 따로 계산합니다. 마스크의 빈 영역은 배경 라벨 `0`으로 채웁니다. 작은 각도 회전은 새로운 밝기값을 만들지는 않지만 디지털 격자 위에서 형태 경계를 바꾸므로, 별도 실험에서 검증한 뒤 사용하는 편이 안전합니다.

테스트 실행:

```powershell
python -m unittest -v test_augmentation.py
```
