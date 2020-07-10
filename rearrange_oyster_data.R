library(tidyverse)
load("oysters_long.RData")

oysterdata_final <- oysters_long %>% 
  mutate_if(
    is.factor, as.character
  ) %>%
  arrange(     
    jar,
    week,
    species,
    treatment,
    replicate,
    desc(individual_id)
  ) %>%
  select(-oyster_number) %>%
  mutate(
    image_id = as.character(image_id),
    surface_area_cm2 = round(surface_area_cm2, digits = 2),
    length_cm = round(length_cm, digits = 2),
    width_cm = round(width_cm, digits = 2),
    surface_area_px2 = round(surface_area_px2, digits = 2),
    length_pixels = round(length_pixels, digits = 2),
    width_pixels = round(width_pixels, digits = 2)
  ) %>%
  pivot_wider(
    names_from = initial.final,
    values_from = c(
      image_id, 
      surface_area_cm2, length_cm, width_cm,
      surface_area_px2, length_pixels, width_pixels,
      pixels_per_cm, pixels2_per_cm2,
      notes
    )#,
  #  values_fn = list
  ) %>%
  select(
    image_id_initial, image_id_final,
    jar,week,species,treatment,replicate,individual_id,
    surface_area_cm2_initial, surface_area_cm2_final,
    length_cm_initial, length_cm_final,
    length_pixels_initial, length_pixels_final,
    surface_area_px2_initial, surface_area_px2_final,
    width_cm_initial, width_cm_final,
    width_pixels_initial, width_pixels_final,
    pixels_per_cm_initial, pixels_per_cm_final,
    pixels2_per_cm2_initial, pixels2_per_cm2_final,
    notes_initial, notes_final
  ) %>%
  unchop(everything())
#view(oysterdata_final)

oysterdata_nodupes <- oysters_long %>%
  distinct(
    jar, week, species, treatment, replicate, individual_id, image_id, .keep_all = T  
  ) %>% 
  mutate_if(
    is.factor, as.character
  ) %>%
  arrange(     
    jar,
    week,
    species,
    treatment,
    replicate,
    desc(individual_id)
  ) %>%
  select(-oyster_number) %>%
  mutate(
    image_id = as.character(image_id),
    surface_area_cm2 = round(surface_area_cm2, digits = 2),
    length_cm = round(length_cm, digits = 2),
    width_cm = round(width_cm, digits = 2),
    surface_area_px2 = round(surface_area_px2, digits = 2),
    length_pixels = round(length_pixels, digits = 2),
    width_pixels = round(width_pixels, digits = 2)
  ) %>%
  pivot_wider(
    names_from = initial.final,
    values_from = c(
      image_id, 
      surface_area_cm2, length_cm, width_cm,
      surface_area_px2, length_pixels, width_pixels,
      pixels_per_cm, pixels2_per_cm2,
      notes
    )
  #values_fn = list
  ) %>%
  select(
    image_id_initial, image_id_final,
    jar,week,species,treatment,replicate,individual_id,
    surface_area_cm2_initial, surface_area_cm2_final,
    length_cm_initial, length_cm_final,
    length_pixels_initial, length_pixels_final,
    surface_area_px2_initial, surface_area_px2_final,
    width_cm_initial, width_cm_final,
    width_pixels_initial, width_pixels_final,
    pixels_per_cm_initial, pixels_per_cm_final,
    pixels2_per_cm2_initial, pixels2_per_cm2_final,
    notes_initial, notes_final
  ) %>%
  unchop(everything())


surface_area_pct_growth <- oysterdata_final %>%
  select(
    image_id_initial, image_id_final,
    jar,week,species,treatment,replicate,individual_id,
    surface_area_cm2_initial, surface_area_cm2_final
  ) %>%
  mutate(
    image_id_initial = as.character(image_id_initial),
    image_id_final = as.character(image_id_final),
    surface_area_cm2_initial = as.numeric(surface_area_cm2_initial),
    surface_area_cm2_final = as.numeric(surface_area_cm2_final)
  ) %>% 
  mutate(
    percent_growth = (surface_area_cm2_final - surface_area_cm2_initial) / surface_area_cm2_initial
  )


surface_area_treatment_species <- surface_area_pct_growth %>%
  group_by(
    species, treatment
  ) %>%
  summarize(
    mean_pct_growth = mean(percent_growth, na.rm = T),
    mean_initial = mean(surface_area_cm2_initial, na.rm = T),
    mean_final = mean(surface_area_cm2_final, na.rm = T),
    n_initial = sum(!is.na(surface_area_cm2_initial)),
    n_final = sum(!is.na(surface_area_cm2_final))
  ) 



surface_area_treatment_species <- surface_area_pct_growth %>%
  group_by(
    species, treatment
  ) %>%
  filter(
    !is.na(surface_area_cm2_final) & !is.na(surface_area_cm2_initial)
  ) %>%
  summarize(
    mean_pct_growth = mean(percent_growth, na.rm = T),
    mean_initial = mean(surface_area_cm2_initial, na.rm = T),
    mean_final = mean(surface_area_cm2_final, na.rm = T),
    sd_initial = sd(surface_area_cm2_initial, na.rm = T),
    sd_final = sd(surface_area_cm2_final, na.rm = T),
    n_initial = sum(!is.na(surface_area_cm2_initial)),
    n_final = sum(!is.na(surface_area_cm2_final)),
    p = t.test(
      x = surface_area_cm2_initial, 
      y = surface_area_cm2_final, 
      mu = 0, 
      var.equal = F, 
      alternative = 'two.sided'
    )$p.value / 2,
    sigdiff = if_else(p < 0.05, TRUE, FALSE)
  ) 

