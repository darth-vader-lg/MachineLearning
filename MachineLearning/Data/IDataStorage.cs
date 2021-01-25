using Microsoft.ML;
using Microsoft.ML.Data;

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
      /// <param name="textLoaderOptions">Eventuali opzioni di caricamento testuale</param>
      /// <returns>L'accesso ai dati</returns>
      IDataAccess LoadData(IMachineLearningContext context, TextLoader.Options textLoaderOptions = default);
      /// <summary>
      /// Salva i dati
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="data">L'accesso ai dati</param>
      /// <param name="textLoaderOptions">Eventuali opzioni di caricamento testuale</param>
      void SaveData(IMachineLearningContext context, IDataView data, TextLoader.Options textLoaderOptions = default);
      #endregion
   }
}
