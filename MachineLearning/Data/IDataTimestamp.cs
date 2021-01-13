using System;

namespace MachineLearning.Data
{
   /// <summary>
   /// Interfaccia per gli oggetti dotati di timestamp
   /// </summary>
   public interface IDataTimestamp
   {
      #region Properties
      /// <summary>
      /// Data e ora dell'oggetto contenente dati
      /// </summary>
      DateTime DataTimestamp { get; }
      #endregion
   }
}
