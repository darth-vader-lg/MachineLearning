using Microsoft.ML;
using System;

namespace MachineLearning.Data
{
   /// <summary>
   /// Classe base per lo storage di dati di tipo binario
   /// </summary>
   [Serializable]
   public abstract class DataStorageBinary : DataStorageBase, IDataStorage
   {
      #region Properties
      /// <summary>
      /// Specifica se mantenere le colonne nascoste nel set di dati
      /// </summary>
      public bool KeepHidden { get; set; }
      #endregion
      #region Methods
      /// <summary>
      /// Carica i dati
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <returns>L'accesso ai dati</returns>
      public virtual IDataAccess LoadData(IMachineLearningContextProvider context) => LoadBinaryData(context);
      /// <summary>
      /// Salva i dati
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="data">L'accesso ai dati</param>
      public abstract void SaveData(IMachineLearningContextProvider context, IDataView data);
      #endregion
   }
}
