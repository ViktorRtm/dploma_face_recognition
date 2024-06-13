--truncate table uniq_face;
CREATE TABLE uniq_face (
    time_created TIMESTAMP NOT NULL DEFAULT current_timestamp,
    id uuid NOT NULL DEFAULT gen_random_uuid(),
    person_id uuid NOT NULL,
    person_vector public.vector,
    face_vector public.vector,
    face_detection_conf FLOAT,
    CONSTRAINT uniq_face_pkey PRIMARY KEY (id)
);
