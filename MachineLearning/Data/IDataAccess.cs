using Microsoft.ML;

namespace MachineLearning.Data
{
   /// <summary>
   /// Interfaccia per l'accesso ai dati
   /// </summary>
   public interface IDataAccess : IDataView, IMachineLearningContextProvider
   {
   }
}
