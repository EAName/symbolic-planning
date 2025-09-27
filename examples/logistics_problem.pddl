(define (problem logistics-problem)
  (:domain logistics)
  (:objects
    apn apt
    tru1
    pos1 pos2
    cit1 cit2
    obj11 obj12
    air1
  )
  (:init
    (airport apn)
    (airport apt)
    (truck tru1)
    (location pos1)
    (location pos2)
    (city cit1)
    (city cit2)
    (package obj11)
    (package obj12)
    (airplane air1)
    (at obj11 pos1)
    (at obj12 pos1)
    (at tru1 pos1)
    (at air1 apn)
  )
  (:goal
    (and
      (at obj11 pos2)
      (at obj12 pos2)
    )
  )
)
