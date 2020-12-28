﻿namespace MachineLearning
{
   /// <summary>
   /// Interfaccia per i provider di dati di training
   /// </summary>
   public interface ITrainingDataProvider
   {
      #region Properties
      /// <summary>
      /// Dati di training
      /// </summary>
      ITrainingData TrainingData { get; }
      #endregion
   }
}
