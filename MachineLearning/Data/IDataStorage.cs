using Microsoft.ML;

namespace MachineLearning.Data
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
      /// <param name="context">Contesto</param>
      /// <returns>L'accesso ai dati</returns>
      IDataAccess LoadData(IMachineLearningContextProvider context);
      /// <summary>
      /// Salva i dati
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="data">L'accesso ai dati</param>
      void SaveData(IMachineLearningContextProvider context, IDataView data);
      #endregion
   }
}
