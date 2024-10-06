안녕하세요우
꺄오가이거

Branch Guideline

- main: 테스트를 마친 소프트웨어가 있는 곳
- feature: 개발 단계에서 기능 단위를 개발하는 곳
  - feature/{name}식으로 개발하고 싶은 기능의 이름을 넣어 브랜치를 생성한다
  - 해당 브랜치 안에서 작업 후 develop에 merge하고 테스트 한다
- develop: feature에서 개발된 기능을 합치는 곳
- release: develop에서 합쳐진 소프트웨어를 QA(테스트)하는 곳
- hotfix: 배포된 소프트웨어의 버그를 고치는 곳
