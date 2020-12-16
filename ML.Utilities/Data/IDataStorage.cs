﻿using Microsoft.ML;

namespace ML.Utilities.Data
{
   /// <summary>
   /// Interfaccia per lo storage di dati
   /// </summary>
   public interface IDataStorage
   {
      #region Methods
      /// <summary>
      /// Carica i dati
      /// </summary>
      /// <param name="mlContext">Contesto di machine learning</param>
      /// <returns>L'accesso ai dati</returns>
      IDataView LoadData(MLContext mlContext);
      /// <summary>
      /// Salva i dati
      /// </summary>
      /// <param name="mlContext">Contesto di machine learning</param>
      /// <param name="data">L'accesso ai dati</param>
      void SaveData(MLContext mlContext, IDataView data);
      #endregion
   }
}
