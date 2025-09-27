(define (problem gripper-problem)
  (:domain gripper)
  (:objects
    rooma roomb
    ball1 ball2
    left right
  )
  (:init
    (room rooma)
    (room roomb)
    (ball ball1)
    (ball ball2)
    (gripper left)
    (gripper right)
    (at-robby rooma)
    (at ball1 rooma)
    (at ball2 rooma)
    (free left)
    (free right)
  )
  (:goal
    (and
      (at ball1 roomb)
      (at ball2 roomb)
    )
  )
)
