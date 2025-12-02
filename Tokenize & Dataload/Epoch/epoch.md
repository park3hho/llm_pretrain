이거 왜이리 용량을 많이 잡아먹냐,,,?


## 쓰로틀링 여부 확인
`nvidia-smi -q -d PERFORMANCE`
```
C:\Users\iwill> nvidia-smi -q -d PERFORMANCE

==============NVSMI LOG==============

Timestamp                                 : Wed Dec  3 06:15:36 2025
Driver Version                            : 576.88
CUDA Version                              : 12.9

Attached GPUs                             : 1
GPU 00000000:01:00.0
    Performance State                     : P0 # 최고 성능 모드
    Clocks Event Reasons
        Idle                              : Not Active
        Applications Clocks Setting       : Not Active #ㅅ 수동 오버/언더 클럭
        SW Power Cap                      : Not Active
        HW Slowdown                       : Not Active 
            HW Thermal Slowdown           : Not Active # 발열 제한 모드
            HW Power Brake Slowdown       : Not Active # 전력 제한 모드
        Sync Boost                        : Not Active
        SW Thermal Slowdown               : Not Active
        Display Clock Setting             : Not Active
    Clocks Event Reasons Counters
        SW Power Capping                  : 3737587 us # 소프트웨어가 전력 때문에 제한한 시간
        Sync Boost                        : 0 us
        SW Thermal Slowdown               : 0 us # 소프트웨어가 온도 때문에 제한한 시간
        HW Thermal Slowdown               : 0 us 
        HW Power Braking                  : 0 us
    Sparse Operation Mode                 : N/A 

```

## GPU Clock & Power Draw
`nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw --format=csv`

```
C:\Users\iwill>nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw --format=csv

clocks.current.sm [MHz], clocks.current.memory [MHz], power.draw [W]
2827 MHz, 14001 MHz, 46.83 W
```