# Segmentation Masking SLAM(SMLAM)
## 프로젝트 목표 
동적 장애물은 SLAM 진행간 keypoint검출에 안좋은 영향을 미치니 segmeation을 통한 masking으로 좋은 결과를 만들어보자
## Roles
- 형승호: carla기반 학습데이터 구축, ORB-SLAM2
- 안현석: custom segmentation model 구현
- 이세라: open source segmentation model 활용
- 서영훈: ORB-SALM2, DynaSLAM과 성능비교

## 개요
Carlr: 학습, 테스트 데이터 생성
Symantic segmentation: custom, paddleSeg(opensource) 개별 학습 및 masking처리
SLAM: masking된 이미지입력에 ORB-SLAM2을 사용해 VSLAM 결과 비교

## 코드
본 repository는 deeplab_v3+, panoptic deeplab을 base 혹은 참조해 구현한 symentic segmentation 모델이다.
다른 파트는 open source를 활용한바가 크기때문에 생략.

# 최종 발표 자료
https://south-spruce-3f7.notion.site/Team3-4ccbeab58fe347898351c121a00b7a14
