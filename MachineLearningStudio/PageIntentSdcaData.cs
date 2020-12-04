﻿using Microsoft.ML.Data;

namespace MachineLearningStudio
{
   /// <summary>
   /// Dati di test algoritmo di previsione intenzioni
   /// </summary>
   public class PageIntentSdcaData
   {
      [LoadColumn(0)]
      public string Intent;
      [LoadColumn(1)]
      public string Sentence;
   }
}
