using System;

namespace MachineLearning
{
   /// <summary>
   /// Interfaccia per gli oggetti dotati di timestamp
   /// </summary>
   public interface ITimestamp
   {
      #region Properties
      /// <summary>
      /// Data e ora dell'oggetto
      /// </summary>
      DateTime Timestamp { get; }
      #endregion
   }
}
