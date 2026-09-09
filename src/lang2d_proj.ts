// ecapa/lang2d 선형 투영 파라미터 (npz mean__/components__/center__/scale__ 추출)
// project: emb(192) → coords = (comp@(emb-mean) - center) / scale

const fromB64 = (b: string) => {
  const raw = atob(b)
  const arr = new Uint8Array(raw.length)
  for (let i = 0; i < raw.length; i++) arr[i] = raw.charCodeAt(i)
  return new Float32Array(arr.buffer)
}
export const PROJ_MEAN = fromB64("vrBuPyTWhsDaomNBHWTkv1y7xj7CUBnBGsXdwGULa78nStvAh7LIQNAChkCwpYtA/MGyQT1xLkAHI2RBQhuIQbIUeEAhVGpBNbkvQENorUD3qTfAKPArwb2u7kD6fCFAUnEIwQL5FMBd8ILAFpajP7dkgcBr0zLBR/qLPyPgk0CN26VAwVCkQG96v0AehRfAO9fbwGM/FcEq+oVAAm4AwPnw7T9rJI1BDtLawCIxFMHhICS/FwdKQCWk7sCNlQbBPXKUQTz5HkDY4mtAobsOwMWtucACvC9An+bbQGd5R8CBF5W/LKUrwZ/ekT90hA5BZd0qwHMEukB1jOu/ZJYSwFt0QkCq/N5A6ClavwnlJsFinx/AyHAHwXaU6cDDR6fA8imTQG4CKMGyg9FAToZewOPFBkCrvkM+J5AAwTEcZUF+LNy/etAmQEr2w0Eli40/06JBQXbDH8B8qshAolhxwJyYhcG2KG1BmBN5wHiHFMFm/3RBDDGCQNfghsHinxlB6D5bQalbsMApjErBz8WmQDSW5T8AHxRBhHOlQBiz/8D5xJ4+32Ctvs9j0cDezdS+ZaCjwC0XY8AZCtLAE6IgQZjTMr+6NCjBZsnMwGzkI0AxrZs/7GYwQfH6g0CndqFAJXfuvlQf8cDhzztAtI50QC3aiUEVjUrBafq1waums8A8mhDBxzn7wCObXkAQb6JAKh8CwTvfNT+K/mrBANXbQALUgz+Yg7m/xkugQITlrMD0MZnAeolLwI1YJ8FJIbK/uLIJQdekjUCH0Vk/EYWYwGaDLUECMOJAGlVvQc51CEEUYANBhb2dQDyCq0DJDC3B46eRPwi3scDqezDBMomUvoowBcFbe1y+a6LzQJZPlL98z45A47ulwECN9r81n4TA68SGwcqboEC5dXRBnjFewC2vIMB+6lRAHMs7wY/TKj8ItSrBkaWjwD3kVUEj37lA8QrhvQL7icFajew+UjEyQBmozz9etEbAZxWmwFgRocErO4VAhMLQveGM4T6uL6m/")
export const PROJ_COMP = fromB64("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")
export const PROJ_CENTER = fromB64("KVwPtwAAAAA=")
export const PROJ_SCALE = 1332.8792724609375

export function projectLang(emb: Float32Array): [number, number] {
  const mean = PROJ_MEAN
  let x = 0
  let y = 0
  for (let i = 0; i < emb.length; i++) {
    const d = emb[i] - mean[i]
    x += PROJ_COMP[i] * d
    y += PROJ_COMP[192 + i] * d
  }
  x = (x - PROJ_CENTER[0]) / PROJ_SCALE
  y = (y - PROJ_CENTER[1]) / PROJ_SCALE
  return [x, y]
}
